#include "astra-sim/common/Logging.hh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace AstraSim {
namespace {

bool is_default_file_logging_disabled(const std::string& log_path) {
    return log_path == "off" || log_path == "none" || log_path == "empty";
}

}  // namespace

std::vector<spdlog::sink_ptr> LoggerFactory::default_sinks;

std::shared_ptr<spdlog::logger> LoggerFactory::get_logger(
    const std::string& logger_name) {
    auto logger = spdlog::get(logger_name);
    if (logger == nullptr) {
        logger = spdlog::create_async<spdlog::sinks::null_sink_mt>(logger_name);
        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::info);
    }
    auto& logger_sinks = logger->sinks();
    for (auto sink : default_sinks) {
        if (std::find(logger_sinks.begin(), logger_sinks.end(), sink) ==
            logger_sinks.end()) {
            logger_sinks.push_back(sink);
        }
    }
    return logger;
}

void LoggerFactory::init(const std::string& log_config_path,
                         const std::string& log_path) {
    if (log_config_path != "empty") {
        // Fail closed with a diagnosis (fix (2026-10-01), ported from the
        // face repo): a missing or malformed logging config used to escape
        // spdlog_setup as an uncaught exception and terminate the process
        // before any logger existed.
        try {
            spdlog_setup::from_file(log_config_path);
        } catch (const std::exception& e) {
            std::cerr << "[Error] (AstraSim) Failed to load logging config file \""
                      << log_config_path << "\": " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    init_default_components(log_path);
}

void LoggerFactory::shutdown(void) {
    default_sinks.clear();
    spdlog::drop_all();
    spdlog::shutdown();
}

void LoggerFactory::init_default_components(const std::string& log_path) {
    auto sink_color_console =
        std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    sink_color_console->set_level(spdlog::level::info);
    default_sinks.push_back(sink_color_console);

    if (is_default_file_logging_disabled(log_path)) {
        spdlog::init_thread_pool(8192, 1);
        spdlog::set_pattern("[%Y-%m-%dT%T%z] [%L] <%n>: %v");
        return;
    }

    std::filesystem::path folderPath(log_path);

    // L9 fix (2026-09-29): --logging-folder may name a NESTED path
    // (run_output/2026-09-29/rank0). create_directory() only creates the
    // FINAL component and throws a bare filesystem_error (-> terminate,
    // without the flag context) when a parent is missing; create_directories
    // builds the whole chain, and a failure is reported with the flag and
    // the system message before the rotating sinks are constructed.
    std::error_code fs_ec;
    std::filesystem::create_directories(folderPath, fs_ec);
    if (fs_ec) {
        std::fprintf(stderr,
                     "[Error] (logging) --logging-folder %s cannot be "
                     "created: %s\n",
                     log_path.c_str(), fs_ec.message().c_str());
        // Fail loud per the repo-wide contract: an unusable log folder must
        // not silently degrade to console-only logging.
        std::exit(EXIT_FAILURE);
    }

    auto sink_rotate_out =
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_path + "/log.log", 1024 * 1024 * 10, 10);
    sink_rotate_out->set_level(spdlog::level::debug);
    default_sinks.push_back(sink_rotate_out);

    auto sink_rotate_err =
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_path + "/err.log", 1024 * 1024 * 10, 10);
    sink_rotate_err->set_level(spdlog::level::err);
    default_sinks.push_back(sink_rotate_err);

    spdlog::init_thread_pool(8192, 1);
    spdlog::set_pattern("[%Y-%m-%dT%T%z] [%L] <%n>: %v");
}

}  // namespace AstraSim
