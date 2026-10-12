#ifndef __COMMON_LOGGING_HH__
#define __COMMON_LOGGING_HH__

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"
#include "spdlog_setup/conf.h"
#include <memory>
#include <string>
#include <vector>

namespace AstraSim {

class LoggerFactory {
  public:
    LoggerFactory() = delete;
    static std::shared_ptr<spdlog::logger> get_logger(
        const std::string& logger_name);
    /// log_conf_path: an spdlog_setup configuration file name, or the
    /// literal "empty" (= no config file; the DEFAULT, and a different
    /// sense of the word than the log_path disable token below).
    /// log_path: the logging FOLDER for the default rotating file sinks;
    /// any of the three disable tokens "off" / "none" / "empty" runs
    /// console-only (no file sink is created). Everything else names the
    /// folder (created on demand). Re-init restarts the default sink set
    /// instead of accumulating it; loggers created BEFORE a re-init keep
    /// their previously installed sinks and will additionally pick up the
    /// new default sinks on their next get_logger -- pair consecutive
    /// init cycles with shutdown() to avoid double console output on such
    /// loggers.
    static void init(const std::string& log_conf_path = "empty",
                     const std::string& log_path = "log");
    static void shutdown(void);

  private:
    static void init_default_components(const std::string& log_path);
    static std::vector<spdlog::sink_ptr> default_sinks;
};

}  // namespace AstraSim

#endif
