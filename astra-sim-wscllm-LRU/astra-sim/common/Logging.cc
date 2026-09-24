#include "astra-sim/common/Logging.hh"
#include <algorithm>
#include <filesystem>

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
        spdlog_setup::from_file(log_config_path);
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

    if (!std::filesystem::exists(folderPath)) {
        std::filesystem::create_directory(folderPath);
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
