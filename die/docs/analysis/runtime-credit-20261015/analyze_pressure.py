#!/usr/bin/env python3
"""Audit actual trace ordering and closed-input completion, no byte/time model."""
import argparse,collections,json,re
from pathlib import Path


def analyze(run):
    nodes={};pattern=re.compile(r'\b(issue|callback),sys->id=(\d+), tick=(\d+), node->id=(\d+), node->name=([^,]+), node->type=(\d+)')
    for match in pattern.finditer((run/'trace-logs/log.log').read_text()):
        kind,rank,tick,node_id,name,node_type=match.groups();key=(int(rank),int(node_id))
        row=nodes.setdefault(key,{'rank':int(rank),'id':int(node_id),'name':name,'type':int(node_type)})
        assert row['name']==name
        assert kind not in row,(key,kind)
        row[kind]=int(tick)
    assert nodes and all('issue' in row and 'callback' in row for row in nodes.values())
    by_rank_name={(row['rank'],row['name']):row for row in nodes.values()}
    capacities=collections.defaultdict(list)
    for row in nodes.values():
        if ('_target_hbm_write' in row['name'] or
                (row['type']==4 and ('_all_layers_' in row['name'] or '_all_passes_' in row['name']))):
            capacities[row['rank']].append(row)
    for rows in capacities.values():rows.sort(key=lambda row:row['id'])
    releases=[];pool_ack_prefixes=set()
    for row in nodes.values():
        match=re.search(r'^(.*_shard\d+)_ack_from_edge(\d+)$',row['name'])
        if match:
            prefix,edge=match.groups();pool_ack_prefixes.add(prefix)
            send=by_rank_name[(row['rank'],prefix+'_send_to_edge'+edge)]
            releases.append({'kind':'remote_store_ack','node':row,'source_send':send})
        match=re.search(r'^(.*)_shard(\d+)_ack_from_rank(\d+)$',row['name'])
        if match:
            prefix,shard,target=match.groups()
            send=by_rank_name[(row['rank'],prefix+f'_send_rank{row["rank"]}_to_rank{target}')]
            releases.append({'kind':'noc_ack','node':row,'source_send':send})
    for row in nodes.values():
        match=re.search(r'^(.*_shard\d+)_(?:remote_store|edge_store)$',row['name'])
        if match and match.group(1) not in pool_ack_prefixes:
            releases.append({'kind':'direct_store','node':row})
    checks=[]
    for entry in releases:
        release=entry['node'];send=entry.get('source_send')
        next_user=next((row for row in capacities[release['rank']] if row['id']>release['id']),None)
        earliest_release=max(release['callback'],send['callback'] if send else 0)
        checks.append({'kind':entry['kind'],'rank':release['rank'],'release_id':release['id'],
            'release_name':release['name'],'release_finish_ns':release['callback'],
            'source_send_finish_ns':send['callback'] if send else None,
            'ack_finishes_after_source_send':release['callback']>=send['callback'] if send else None,
            'next_capacity_user':next_user,
            'capacity_reuse_waits_for_both':next_user['issue']>=earliest_release if next_user else None})
    decisions=[json.loads(line) for line in (run/'bridge/online_decision_log.jsonl').read_text().splitlines()]
    completion=[row for row in decisions if row.get('kind')=='completion']
    assert len(completion)==24 and len({row['request_id'] for row in completion})==24
    assert checks and any(row['next_capacity_user'] for row in checks)
    assert all(row['capacity_reuse_waits_for_both'] is not False for row in checks)
    log=(run/'cpp.log').read_text()
    audit=re.search(r'terminal_total=(\d+) terminal_success=(\d+) terminal_skipped=(\d+) terminal_other=(\d+)',log)
    assert audit and audit[1]==audit[2] and audit[3]=='0' and audit[4]=='0'
    return {'run':str(run),'unique_trace_nodes':len(nodes),'completed_requests':len(completion),
        'terminal_success':int(audit[2]),'history_actions':dict(collections.Counter(row.get('decision',{}).get('history_action') for row in decisions if row.get('kind')=='prefill')),
        'release_counts':dict(collections.Counter(row['kind'] for row in checks)),
        'actual_remote_store_nodes':sum(bool(re.search(r'_shard\d+_(?:remote_store|edge_store)$',row['name'])) for row in nodes.values()),
        'actual_restore_write_nodes':sum(row['name'].endswith('_target_hbm_write') for row in nodes.values()),
        'source_reuse_checks':sum(row['next_capacity_user'] is not None for row in checks),
        'checks':checks}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.run)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='checks'},sort_keys=True))


if __name__=='__main__':main()
