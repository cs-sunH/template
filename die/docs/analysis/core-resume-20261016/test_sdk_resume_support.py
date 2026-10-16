#!/usr/bin/env python3
"""Small filesystem-fixture retention checks; no CMake/build/test application run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from sdk_resume_support import artifact_snapshot, cache_validation, retention, sha, write_json


class ResumeIdentityTest(unittest.TestCase):
    def setUp(self):
        work = Path(__file__).resolve().parents[4]/'work'
        self.temp = tempfile.TemporaryDirectory(prefix='sdk-resume-unit-', dir=work)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.command = {'label':'phase','command':['fixture-tool','--argument','value'],
            'cwd':str(self.root),'environment':{},'expected':None,'requires':[],'new_build':False}
        self.phase = self.root/'phase';self.phase.mkdir()
        (self.phase/'run.log').write_text('actual completed fixture log\n')
        self.old = {**self.command,'status':'failed','return_code':8,'source_digest':'source',
            'snapshot_manifest_sha256':'manifest','runner_sha256':'1'*64,
            'log_sha256':sha(self.phase/'run.log'),'libraries':{}}
        write_json(self.phase/'identity.json',self.old)

    def check(self, command=None, digest='source', manifest='manifest'):
        return retention(self.root, command or self.command, digest, manifest)

    def test_failed_is_retained_as_failed(self):
        row = self.check();self.assertTrue(row['retain']);self.assertEqual(row['original_status'],'failed')
        self.assertEqual(row['old_record']['return_code'],8)

    def test_command_cwd_env_changes_are_rejected(self):
        for key, value in [('command',['different-tool']),('cwd','/different'),('environment',{'MODE':'different'})]:
            with self.subTest(key=key):
                command=copy.deepcopy(self.command);command[key]=value
                self.assertFalse(self.check(command)['retain'])

    def test_source_or_manifest_changes_are_rejected(self):
        self.assertFalse(self.check(digest='new-source')['retain'])
        self.assertFalse(self.check(manifest='new-manifest')['retain'])

    def test_log_tampering_is_rejected(self):
        (self.phase/'run.log').write_text('not original\n')
        self.assertFalse(self.check()['retain'])

    def test_missing_final_identity_is_incomplete_even_with_log(self):
        (self.phase/'identity.json').unlink()
        row=self.check();self.assertFalse(row['retain'])
        self.assertEqual(row['reason'],'incomplete_no_final_identity')
        self.assertFalse(row['started_present'])

    def test_nonterminal_record_is_not_retained(self):
        self.old.update(status='interrupted',return_code=None);write_json(self.phase/'identity.json',self.old)
        self.assertFalse(self.check()['retain'])

    def test_configure_requires_matching_live_cache(self):
        build=self.root/'build';build.mkdir();source=self.root/'source';source.mkdir()
        command={**self.command,'new_build':True,'command':['cmake','-S',str(source),'-B',str(build),'-G','Unix Makefiles','-DCMAKE_BUILD_TYPE=Release']}
        self.old.update(command=command['command'],new_build=True,status='passed',return_code=0)
        write_json(self.phase/'identity.json',self.old)
        self.assertFalse(self.check(command)['retain'])
        cache=build/'CMakeCache.txt';cache.write_text('CMAKE_HOME_DIRECTORY:INTERNAL='+str(source)+'\nCMAKE_GENERATOR:INTERNAL=Unix Makefiles\nCMAKE_BUILD_TYPE:STRING=Release\n')
        self.assertTrue(self.check(command)['retain'])
        cache.write_text(cache.read_text().replace('Release','Debug'))
        self.assertFalse(self.check(command)['retain'])

    def test_actual_target_digest_and_drift_are_recorded(self):
        build=self.root/'build';build.mkdir();binary=build/'ActualTarget';binary.write_text('fixture artifact A')
        (build/'CTestTestfile.cmake').write_text('add_test(ActualTarget "'+str(binary)+'")\n')
        command={**self.command,'command':['ctest','--test-dir',str(build),'-R','^(ActualTarget)$']}
        before=artifact_snapshot(command,[command])
        self.assertEqual(before['targets']['ActualTarget']['files'][str(binary)]['sha256'],sha(binary))
        self.old.update(command=command['command'],artifacts_after=before)
        write_json(self.phase/'identity.json',self.old)
        self.assertTrue(self.check(command)['retain'])
        binary.write_text('fixture artifact B')
        after=artifact_snapshot(command,[command]);self.assertNotEqual(before,after)
        self.assertFalse(self.check(command)['retain'])

if __name__=='__main__':unittest.main(verbosity=2)
