#include <string>
#include <cassert>
#include <cmath>
#include <fstream>
#include <memory>
// booksim include
#include "include/routefunc.hpp"
#include "include/booksim_config.hpp"
#include "include/trafficmanager.hpp"
#include "include/globals.hpp"
#include "include/flit.hpp"
#include "networks/network.hpp"
#include "routers/router.hpp"
#include "routers/mc_router.hpp"
// atlasim include
#include "noc/noc_wrapper.h"


// printing activity factor
bool gPrintActivity;
int gK; //radix
int gN; //dimension
int gC; //concentration
int gNodes;
//generate nocviewer trace
bool gTrace;
ostream * gWatchOut;

// Well ... old troublesomes from booksim2
TrafficManager* trafficManager = NULL;


namespace atlasim {

// Static member initialization
int NoCWrapper::instance_count = 0;

NoCWrapper::NoCWrapper( BookSimConfig _config, PCNInterfaceSet _send_queues, PCNInterfaceSet _received_queues) {
    instance_count++;
    config = _config;
    InitializeRoutingMap(config);

    gPrintActivity = (config.GetInt("print_activity") > 0);
    gTrace = (config.GetInt("viewer_trace") > 0);

    string watch_out_file = config.GetStr( "watch_out" );
    if(watch_out_file == "") {
        gWatchOut = NULL;
    } else if(watch_out_file == "-") {
        gWatchOut = &cout;
    } else {
        if(gWatchOut && gWatchOut != &cout) {
            delete gWatchOut;
        }
        gWatchOut = new ofstream(watch_out_file.c_str());
    }

    // Setup Nets
    vector<Network *> net;
    int subnets = config.GetInt("subnets");
    if (subnets != 1) {
        throw "The number of subnets is not 1 !!";
    }
    net.resize(subnets);
    for (int i = 0; i < subnets; ++i) {
        ostringstream name;
        name << "network_" << i;
        net[i] = Network::New( config, name.str());
    }

    traffic_manager = std::shared_ptr<TrafficManager>(TrafficManager::New( config, net ));
    traffic_manager->SetupSim(_send_queues, _received_queues);
    trafficManager = traffic_manager.get();
}


NoCWrapper::~NoCWrapper() {
    instance_count--;

    traffic_manager.reset();  // triggers ~TrafficManager(), which checks instance_count

    if(instance_count == 0) {
        // Last instance — safe to clean up all global state
        trafficManager = NULL;
        gWatchOut = NULL;
        ::gRoutingFunctionMap.clear();
    }
}


void NoCWrapper::reset_execution_status() {
    prev_clk.emplace_back(clk);
}


void NoCWrapper::tick() {
    clk++;
    trafficManager = traffic_manager.get();
    traffic_manager->_Step();

    if(clk % 20000 == 0) {
        Flit::GC();
    }
}


std::vector<double> NoCWrapper::router_conflict_factors() {
    // We assume only one net
    vector<vector<Router*> > routers = dynamic_cast<SpatialTrafficManager*>(traffic_manager.get())->getRouters();
    assert(routers.size() == 1);
    vector<Router*> first_net_routers = routers.front();

    std::vector<double> ret(first_net_routers.size());
    for (int i = 0; i < first_net_routers.size(); ++i) {
        ret[i] = dynamic_cast<MCRouter*>(first_net_routers[i])->ConflictFactor();
    }
    return ret;
}


void NoCWrapper::display_stats(std::ostream& os) {
    traffic_manager->_DisplayRemaining(os);
}


}