/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include <cstdlib>
#include <cxxopts/cxxopts.hpp>
#include <iostream>
#include <typeinfo>

namespace AstraSimAnalytical {

/**
 * CmdLineParser parses command line arguments using cxxopts
 * for ASTRA-sim input parameters.
 */
class CmdLineParser {
  public:
    /**
     * Constructor.
     *
     * @param argv0 argv[0] from main function
     */
    explicit CmdLineParser(const char* argv0) noexcept;

    /**
     * Parse the command line arguments.
     *
     * @param argc argc from main function
     * @param argv argv from main function
     */
    void parse(int argc, char* argv[]) noexcept;

    /**
     * Get the value of the option in type T with the given name.
     *
     * @tparam T type of the argument value
     * @param name name of the argument
     * @return read argument value in type T
     */
    template <typename T> T get(const std::string& name) const noexcept {
        try {
            // try retrieving options
            return parsed[name].as<T>();
        } catch (const cxxopts::OptionException& e) {
            // error occured
            std::cerr << "[Error] (AstraSim/analytical/common) "
                      << "Error parsing options: " << e.what() << std::endl;
            exit(-1);
        } catch (const std::bad_cast& e) {
            // A type-mismatched get<T>() throws std::bad_cast (the base of
            // cxxopts' std::bad_any_cast), which is NOT an OptionException:
            // it used to escape this noexcept function and die as a bare
            // std::terminate. Fail closed the same way parse() does (M5-03
            // fix style, 2026-09-26; bad_cast arm added 2026-09-29).
            std::cerr << "[Error] (AstraSim/analytical/common) "
                      << "Error parsing options: " << e.what() << std::endl;
            exit(-1);
        }
    }

  private:
    /// cxxopts options
    cxxopts::Options options;

    /// cxxopts parsed results
    cxxopts::ParseResult parsed;

    /**
     * Define the options for cxxopts.
     */
    void define_options() noexcept;
};

}  // namespace AstraSimAnalytical
