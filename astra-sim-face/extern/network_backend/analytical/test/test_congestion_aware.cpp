/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/EventQueue.h"
#include "common/NetworkParser.h"
#include "common/Type.h"
#include "congestion_aware/Chunk.h"
#include "congestion_aware/Helper.h"
#include "congestion_aware/fluid/FluidScheduler.h"
#include <gtest/gtest.h>
#include <vector>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

class TestNetworkAnalyticalCongestionAware : public ::testing::Test {
  protected:
    void SetUp() override {
        // set event queue
        event_queue = std::make_shared<EventQueue>();
        Topology::set_event_queue(event_queue);

        // set chunk size
        chunk_size = 1'048'576;  // 1 MB
    }

    std::shared_ptr<EventQueue> event_queue;

    static void callback(void* const arg) {}

    struct ArrivalTracker {
        EventQueue* event_queue;
        std::vector<EventTime> arrival_times;
    };

    static void record_arrival(void* const arg) {
        auto* const tracker = static_cast<ArrivalTracker*>(arg);
        tracker->arrival_times.push_back(tracker->event_queue->get_current_time());
    }

    std::shared_ptr<FluidScheduler> make_fluid_scheduler(const std::shared_ptr<Topology>& topology) {
        return std::make_shared<FluidScheduler>(
            event_queue, topology->get_directed_links(), 20'000, 500'000, 100'000);
    }

    void run_simulation() {
        while (!event_queue->finished()) {
            event_queue->proceed();
        }
    }

    ChunkSize chunk_size;
};

TEST_F(TestNetworkAnalyticalCongestionAware, Ring) {
    /// setup
    const auto network_parser = NetworkParser("../../input/Ring.yml");
    const auto topology = construct_topology(network_parser);

    /// message settings
    auto route = topology->route(1, 4);
    auto chunk = std::make_unique<Chunk>(chunk_size, route, callback, nullptr);

    // send a chunk
    topology->send(std::move(chunk));

    /// Run simulation
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    /// test
    const auto simulation_time = event_queue->get_current_time();
    EXPECT_EQ(simulation_time, 60'093);
}

TEST_F(TestNetworkAnalyticalCongestionAware, FullyConnected) {
    /// setup
    const auto network_parser = NetworkParser("../../input/FullyConnected.yml");
    const auto topology = construct_topology(network_parser);

    /// message settings
    auto route = topology->route(1, 4);
    auto chunk = std::make_unique<Chunk>(chunk_size, route, callback, nullptr);

    // send a chunk
    topology->send(std::move(chunk));

    /// Run simulation
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    /// test
    const auto simulation_time = event_queue->get_current_time();
    EXPECT_EQ(simulation_time, 20'031);
}

TEST_F(TestNetworkAnalyticalCongestionAware, Switch) {
    /// setup
    const auto network_parser = NetworkParser("../../input/Switch.yml");
    const auto topology = construct_topology(network_parser);

    /// message settings
    auto route = topology->route(1, 4);
    auto chunk = std::make_unique<Chunk>(chunk_size, route, callback, nullptr);

    // send a chunk
    topology->send(std::move(chunk));

    /// Run simulation
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    /// test
    const auto simulation_time = event_queue->get_current_time();
    EXPECT_EQ(simulation_time, 40'062);
}

TEST_F(TestNetworkAnalyticalCongestionAware, Mesh2DRoute) {
    /// setup
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);

    /// test
    EXPECT_EQ(topology->get_dims_count(), 2);
    EXPECT_EQ(topology->get_npus_count(), 16);

    const auto route = topology->route(0, 15);
    const auto expected_route = std::vector<DeviceId>({0, 1, 2, 3, 7, 11, 15});
    ASSERT_EQ(route.size(), expected_route.size());

    auto route_iter = route.begin();
    for (const auto expected_device_id : expected_route) {
        EXPECT_EQ((*route_iter)->get_id(), expected_device_id);
        route_iter++;
    }
}

TEST_F(TestNetworkAnalyticalCongestionAware, Mesh2D) {
    /// setup
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);

    /// message settings
    auto route = topology->route(0, 15);
    auto chunk = std::make_unique<Chunk>(chunk_size, route, callback, nullptr);

    // send a chunk
    topology->send(std::move(chunk));

    /// Run simulation
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    /// test
    const auto simulation_time = event_queue->get_current_time();
    EXPECT_EQ(simulation_time, 120'186);
}

TEST_F(TestNetworkAnalyticalCongestionAware, Mesh2DLinkCount) {
    /// setup
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);

    /// test
    EXPECT_EQ(topology->get_links_count(), 48);
}

TEST_F(TestNetworkAnalyticalCongestionAware, FluidResourceDefaultsAreParsed) {
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    EXPECT_EQ(network_parser.get_fluid_max_active_flows(), 20'000);
    EXPECT_EQ(network_parser.get_fluid_max_route_memberships(), 500'000);
    EXPECT_EQ(network_parser.get_progress_report_event_interval(), 100'000);
}

TEST_F(TestNetworkAnalyticalCongestionAware, ExplicitFluidConfigurationIsParsed) {
    const auto network_parser = NetworkParser("../../input/Mesh2DFluid.yml");
    EXPECT_EQ(network_parser.get_fluid_max_active_flows(), 20'000);
    EXPECT_EQ(network_parser.get_fluid_max_route_memberships(), 500'000);
}

TEST_F(TestNetworkAnalyticalCongestionAware, FluidRouteIsSharedAndUsesDirectedLinks) {
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);

    const auto forward = topology->fluid_route(0, 3);
    const auto forward_again = topology->fluid_route(0, 3);
    const auto reverse = topology->fluid_route(3, 0);

    EXPECT_EQ(forward.get(), forward_again.get());
    ASSERT_EQ(forward->link_ids.size(), 3);
    EXPECT_EQ(forward->propagation_latency_ns, 1'500);
    for (size_t i = 0; i < forward->link_ids.size(); ++i) {
        EXPECT_NE(forward->link_ids[i], reverse->link_ids[reverse->link_ids.size() - 1 - i]);
    }
}

TEST_F(TestNetworkAnalyticalCongestionAware, FluidPipelinesMultiHopMessage) {
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);
    const auto scheduler = make_fluid_scheduler(topology);
    auto tracker = ArrivalTracker{event_queue.get(), {}};

    scheduler->start_flow(chunk_size, topology->fluid_route(0, 15), record_arrival, &tracker);
    scheduler->flush_pending_starts();
    scheduler->mark_event_loop_started();
    run_simulation();

    ASSERT_EQ(tracker.arrival_times.size(), 1);
    EXPECT_EQ(tracker.arrival_times[0], 22'532);
    EXPECT_EQ(scheduler->get_total_started_flows(), 1);
    EXPECT_EQ(scheduler->get_total_completed_flows(), 1);
    EXPECT_EQ(scheduler->get_active_flow_count(), 0);
    EXPECT_EQ(scheduler->get_active_route_memberships(), 0);
}

TEST_F(TestNetworkAnalyticalCongestionAware, FluidHundredGiBMessageStillUsesOneFlow) {
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);
    const auto scheduler = make_fluid_scheduler(topology);
    auto tracker = ArrivalTracker{event_queue.get(), {}};
    constexpr ChunkSize hundred_gib = 100ULL * 1024 * 1024 * 1024;

    scheduler->start_flow(hundred_gib, topology->fluid_route(0, 15), record_arrival, &tracker);
    scheduler->flush_pending_starts();
    scheduler->mark_event_loop_started();
    run_simulation();

    ASSERT_EQ(tracker.arrival_times.size(), 1);
    EXPECT_EQ(tracker.arrival_times[0], 2'000'003'001);
    EXPECT_EQ(scheduler->get_total_started_flows(), 1);
    EXPECT_EQ(scheduler->get_total_completed_flows(), 1);
    EXPECT_EQ(scheduler->get_completion_heap_size(), 0);
}

TEST_F(TestNetworkAnalyticalCongestionAware, FluidEqualSharesOneDirectedRoute) {
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);
    const auto scheduler = make_fluid_scheduler(topology);
    auto tracker = ArrivalTracker{event_queue.get(), {}};

    scheduler->start_flow(chunk_size, topology->fluid_route(0, 3), record_arrival, &tracker);
    scheduler->start_flow(chunk_size, topology->fluid_route(0, 3), record_arrival, &tracker);
    scheduler->flush_pending_starts();
    scheduler->mark_event_loop_started();
    run_simulation();

    ASSERT_EQ(tracker.arrival_times.size(), 2);
    EXPECT_EQ(tracker.arrival_times[0], 40'563);
    EXPECT_EQ(tracker.arrival_times[1], 40'563);
    EXPECT_EQ(scheduler->get_total_completed_flows(), 2);
}

TEST_F(TestNetworkAnalyticalCongestionAware, FluidOppositeDirectionsAreFullDuplex) {
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);
    const auto scheduler = make_fluid_scheduler(topology);
    auto tracker = ArrivalTracker{event_queue.get(), {}};

    scheduler->start_flow(chunk_size, topology->fluid_route(0, 3), record_arrival, &tracker);
    scheduler->start_flow(chunk_size, topology->fluid_route(3, 0), record_arrival, &tracker);
    scheduler->flush_pending_starts();
    scheduler->mark_event_loop_started();
    run_simulation();

    ASSERT_EQ(tracker.arrival_times.size(), 2);
    EXPECT_EQ(tracker.arrival_times[0], 21'032);
    EXPECT_EQ(tracker.arrival_times[1], 21'032);
}

TEST_F(TestNetworkAnalyticalCongestionAware, FluidRateRecoversAfterSmallerFlowCompletes) {
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);
    const auto scheduler = make_fluid_scheduler(topology);
    auto tracker = ArrivalTracker{event_queue.get(), {}};

    scheduler->start_flow(chunk_size, topology->fluid_route(0, 3), record_arrival, &tracker);
    scheduler->start_flow(2 * chunk_size, topology->fluid_route(0, 3), record_arrival, &tracker);
    scheduler->flush_pending_starts();
    scheduler->mark_event_loop_started();
    run_simulation();

    ASSERT_EQ(tracker.arrival_times.size(), 2);
    EXPECT_EQ(tracker.arrival_times[0], 40'563);
    EXPECT_EQ(tracker.arrival_times[1], 60'095);
}

TEST_F(TestNetworkAnalyticalCongestionAware, Mesh2DSharedDirectedLinkContention) {
    /// setup
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);
    auto tracker = ArrivalTracker{event_queue.get(), {}};

    /// message settings
    auto route_1 = topology->route(0, 3);
    auto route_2 = topology->route(0, 3);
    auto chunk_1 = std::make_unique<Chunk>(chunk_size, route_1, record_arrival, &tracker);
    auto chunk_2 = std::make_unique<Chunk>(chunk_size, route_2, record_arrival, &tracker);

    // send two chunks that share every directed link
    topology->send(std::move(chunk_1));
    topology->send(std::move(chunk_2));

    /// Run simulation
    run_simulation();

    /// test
    ASSERT_EQ(tracker.arrival_times.size(), 2);
    EXPECT_EQ(tracker.arrival_times[0], 60'093);
    EXPECT_EQ(tracker.arrival_times[1], 79'624);
}

TEST_F(TestNetworkAnalyticalCongestionAware, Mesh2DReverseDirectionFullDuplex) {
    /// setup
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);
    auto tracker = ArrivalTracker{event_queue.get(), {}};

    /// message settings
    auto route_1 = topology->route(0, 3);
    auto route_2 = topology->route(3, 0);
    auto chunk_1 = std::make_unique<Chunk>(chunk_size, route_1, record_arrival, &tracker);
    auto chunk_2 = std::make_unique<Chunk>(chunk_size, route_2, record_arrival, &tracker);

    // send chunks over opposite directed links
    topology->send(std::move(chunk_1));
    topology->send(std::move(chunk_2));

    /// Run simulation
    run_simulation();

    /// test
    ASSERT_EQ(tracker.arrival_times.size(), 2);
    EXPECT_EQ(tracker.arrival_times[0], 60'093);
    EXPECT_EQ(tracker.arrival_times[1], 60'093);
}

TEST_F(TestNetworkAnalyticalCongestionAware, Mesh2DDisjointPathsDoNotContend) {
    /// setup
    const auto network_parser = NetworkParser("../../input/Mesh2D.yml");
    const auto topology = construct_topology(network_parser);
    auto tracker = ArrivalTracker{event_queue.get(), {}};

    /// message settings
    auto route_1 = topology->route(0, 3);
    auto route_2 = topology->route(4, 7);
    auto chunk_1 = std::make_unique<Chunk>(chunk_size, route_1, record_arrival, &tracker);
    auto chunk_2 = std::make_unique<Chunk>(chunk_size, route_2, record_arrival, &tracker);

    // send chunks over disjoint directed links
    topology->send(std::move(chunk_1));
    topology->send(std::move(chunk_2));

    /// Run simulation
    run_simulation();

    /// test
    ASSERT_EQ(tracker.arrival_times.size(), 2);
    EXPECT_EQ(tracker.arrival_times[0], 60'093);
    EXPECT_EQ(tracker.arrival_times[1], 60'093);
}

TEST_F(TestNetworkAnalyticalCongestionAware, AllGatherOnRing) {
    /// setup
    const auto network_parser = NetworkParser("../../input/Ring.yml");
    const auto topology = construct_topology(network_parser);
    const auto npus_count = topology->get_npus_count();

    /// message settings
    const auto chunk_size = 1'048'576;  // 1 MB

    /// Run All-Gather
    for (int i = 0; i < npus_count; i++) {
        for (int j = 0; j < npus_count; j++) {
            if (i == j) {
                continue;
            }

            // crate a chunk
            auto route = topology->route(i, j);
            auto* event_queue_ptr = static_cast<void*>(event_queue.get());
            auto chunk = std::make_unique<Chunk>(chunk_size, route, callback, nullptr);

            // send a chunk
            topology->send(std::move(chunk));
        }
    }

    /// Run simulation
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    /// test
    const auto simulation_time = event_queue->get_current_time();
    EXPECT_EQ(simulation_time, 704'116);
}
