#include "astra-sim/common/Logging.hh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace AstraSim {
namespace {

bool is_default_file_logging_disabled(const std::string& log_path) {
    return log_path == "off" || log_path == "none" || log_path == "empty";
}

}  // namespace

std::vector<spdlog::sink_ptr> LoggerFactory::default_sinks;

// Loggers this factory has attached default sinks to (2026-10-01 R2 audit):
// re-init must DETACH the old default sinks from them, not just clear the
// vector -- otherwise those loggers keep writing to the retired console /
// rotating sinks forever (duplicated lines across generations).
std::vector<std::string> LoggerFactory::default_sink_attached_loggers;

std::shared_ptr<spdlog::logger> LoggerFactory::get_logger(
    const std::string& logger_name) {
    constexpr bool ENABLE_DEFAULT_SINK_FOR_OTHER_LOGGERS = true;
    auto logger = spdlog::get(logger_name);
    if (logger == nullptr) {
        logger = spdlog::create_async<spdlog::sinks::null_sink_mt>(logger_name);
        logger->set_level(spdlog::level::trace);
        logger->flush_on(spdlog::level::info);
    }
    if constexpr (!ENABLE_DEFAULT_SINK_FOR_OTHER_LOGGERS) {
        return logger;
    }
    auto& logger_sinks = logger->sinks();
    bool attached_any = false;
    for (auto sink : default_sinks) {
        if (std::find(logger_sinks.begin(), logger_sinks.end(), sink) ==
            logger_sinks.end()) {
            logger_sinks.push_back(sink);
            attached_any = true;
        }
    }
    if (attached_any &&
        std::find(default_sink_attached_loggers.begin(),
                  default_sink_attached_loggers.end(),
                  logger_name) == default_sink_attached_loggers.end()) {
        default_sink_attached_loggers.push_back(logger_name);
    }
    return logger;
}

void LoggerFactory::init(const std::string& log_config_path,
                         const std::string& log_path) {
    // Re-initialization hygiene (2026-10-01 audit): test binaries build
    // several fixtures in one process and each calls init again; without
    // this clear, every repeat APPENDED another console/rotating sink to
    // the static default_sinks vector (duplicated log lines) and
    // re-attached stale sinks to already-configured loggers. Dropping the
    // old defaults first -- from every logger they were attached to and
    // from the vector itself -- makes init idempotent.
    const std::vector<spdlog::sink_ptr> retired_defaults(default_sinks);
    for (const std::string& logger_name : default_sink_attached_loggers) {
        auto logger = spdlog::get(logger_name);
        if (logger == nullptr) {
            continue;
        }
        auto& logger_sinks = logger->sinks();
        logger_sinks.erase(
            std::remove_if(logger_sinks.begin(), logger_sinks.end(),
                           [&retired_defaults](const spdlog::sink_ptr& s) {
                               return std::find(retired_defaults.begin(),
                                                retired_defaults.end(),
                                                s) != retired_defaults.end();
                           }),
            logger_sinks.end());
    }
    default_sink_attached_loggers.clear();
    default_sinks.clear();
    if (log_config_path != "empty") {
        spdlog_setup::from_file(log_config_path);
    }
    init_default_components(log_path);
}

void LoggerFactory::shutdown(void) {
    default_sink_attached_loggers.clear();
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
