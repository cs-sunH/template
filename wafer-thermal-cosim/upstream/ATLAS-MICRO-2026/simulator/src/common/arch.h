#ifndef _ATLASIM_ARCH_H
#define _ATLASIM_ARCH_H

#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <yaml-cpp/yaml.h>


namespace atlasim {


inline double read_optional_double(
    const YAML::Node& config,
    const std::string& key,
    const std::string& config_name
) {
    try {
        return config[key].as<double>();
    }
    catch(const std::exception& e) {
        std::cerr << "[Warning] " << config_name << " config does not provide " << key << " information." << std::endl;
        return 0.0;
    }
}


struct ControllerConfig {
    double power = 0.0; // W
    double area = 0.0;

    ControllerConfig() { }
    ControllerConfig(const YAML::Node config) {
        power = read_optional_double(config, "power", "Controller");
        area = read_optional_double(config, "area", "Controller");
    }
};


struct MatrixConfig {
    int mac_num = 0;
    double power = 0.0; // W
    double area = 0.0;

    MatrixConfig() { }
    MatrixConfig(const YAML::Node config) {
        mac_num = config["mac_num"].as<int>();
        power = read_optional_double(config, "power", "Matrix");
        area = read_optional_double(config, "area", "Matrix");
    }
};


struct VectorConfig {
    int vec_num = 0;
    double power = 0.0; // W
    double area = 0.0;

    VectorConfig() { }
    VectorConfig(const YAML::Node config) {
        vec_num = config["vec_num"].as<int>();
        power = read_optional_double(config, "power", "Vector");
        area = read_optional_double(config, "area", "Vector");
    }
};


struct BufferConfig {
    int buffer_size = 0; // KB
    // Assume the buffer has independent on-chip read/write ports
    double read_bw = 0.0; // Byte/cycle
    double write_bw = 0.0; // Byte/cycle
    double power = 0.0; // W
    double area = 0.0;

    BufferConfig() { }
    BufferConfig(const YAML::Node config) {
        buffer_size = config["buffer_size"].as<int>();
        read_bw = config["read_bw"].as<double>();
        write_bw = config["write_bw"].as<double>();
        power = read_optional_double(config, "power", "Buffer");
        area = read_optional_double(config, "area", "Buffer");
    }
};


struct DRAMConfig {
    std::string config_path;
    double power = 0.0; // W
    double area = 0.0;

    DRAMConfig() { }
    DRAMConfig(const YAML::Node config) {
        config_path = config["config_path"].as<std::string>();
        power = read_optional_double(config, "power", "DRAM");
        area = read_optional_double(config, "area", "DRAM");
    }
};


struct NoCConfig {
    std::string topology;
    std::string config_path;   // booksim config path
    int flit_size = 0;         // unit: Byte
    double power = 0.0;        // W
    double area = 0.0;

    NoCConfig() { }
    NoCConfig(const YAML::Node config) {
        topology = config["topology"].as<std::string>();
        config_path = config["config_path"].as<std::string>();
        flit_size = config["flit_size"].as<int>();
        power = read_optional_double(config, "power", "NoC");
        area = read_optional_double(config, "area", "NoC");
    }
};


struct ChipConfig {
    // Global config
    double frequency = 0.0; // MHz
    int core_num = 0;
    // On-chip config
    ControllerConfig controller_config;
    MatrixConfig matrix_config;
    VectorConfig vector_config;
    BufferConfig buffer_config;
    // DRAM config
    DRAMConfig dram_config;
    // NoC config
    bool has_noc = false;
    NoCConfig noc_config;

    ChipConfig() { }
    ChipConfig(const YAML::Node config) {
        frequency = config["architecture"]["frequency"].as<double>();
        core_num = config["architecture"]["core_num"].as<int>();
        // On-chip config
        controller_config = ControllerConfig(config["architecture"]["core"]["controller"]);
        matrix_config = MatrixConfig(config["architecture"]["core"]["matrix"]);
        vector_config = VectorConfig(config["architecture"]["core"]["vector"]);
        buffer_config = BufferConfig(config["architecture"]["core"]["buffer"]);
        // DRAM config
        dram_config = DRAMConfig(config["architecture"]["dram"]);
        // NoC config
        YAML::Node noc_node = config["architecture"]["noc"];
        has_noc = noc_node && !noc_node.IsNull();
        if(has_noc) {
            noc_config = NoCConfig(noc_node);
        }
    }

    void display_stats(std::ostream& os=std::cout, int indent=0) {
        os << std::setw(indent) << "Arch Config:" << std::endl;
        os << std::setw(indent) << "    Frequency (MHz): " << frequency << std::endl;
        os << std::setw(indent) << "    Core Number: " << core_num << std::endl;
        
        os << std::setw(indent) << "    Core Spec:" << std::endl;
        os << std::setw(indent) << "    (1) Controller:" << std::endl;
        os << std::setw(indent) << "        Power (W): " << controller_config.power << std::endl;
        os << std::setw(indent) << "        Area: " << controller_config.area << std::endl;

        os << std::setw(indent) << "    (2) Matrix:" << std::endl;
        os << std::setw(indent) << "        MAC Number: " << matrix_config.mac_num << std::endl;
        os << std::setw(indent) << "        Power (W): " << matrix_config.power << std::endl;
        os << std::setw(indent) << "        Area: " << matrix_config.area << std::endl;

        os << std::setw(indent) << "    (3) Vector:" << std::endl;
        os << std::setw(indent) << "        Vector Number: " << vector_config.vec_num << std::endl;
        os << std::setw(indent) << "        Power (W): " << vector_config.power << std::endl;
        os << std::setw(indent) << "        Area: " << vector_config.area << std::endl;

        os << std::setw(indent) << "    (4) Buffer:" << std::endl;
        os << std::setw(indent) << "        Buffer Size (KB): " << buffer_config.buffer_size << std::endl;
        os << std::setw(indent) << "        Read BW (Byte/cycle): " << buffer_config.read_bw << std::endl;
        os << std::setw(indent) << "        Write BW (Byte/cycle): " << buffer_config.write_bw << std::endl;
        os << std::setw(indent) << "        Power (W): " << buffer_config.power << std::endl;
        os << std::setw(indent) << "        Area: " << buffer_config.area << std::endl;

        os << std::setw(indent) << "    (5) DRAM:" << std::endl;
        os << std::setw(indent) << "        Ramulator Config Path: " << dram_config.config_path << std::endl;
        os << std::setw(indent) << "        Power (W): " << dram_config.power << std::endl;
        os << std::setw(indent) << "        Area: " << dram_config.area << std::endl;

        os << std::setw(indent) << "    (6) NoC Spec:" << std::endl;
        os << std::setw(indent) << "        Has NoC: " << has_noc << std::endl;
        if(has_noc) {
            os << std::setw(indent) << "        Topology: " << noc_config.topology << std::endl;
            os << std::setw(indent) << "        Booksim Config Path: " << noc_config.config_path << std::endl;
            os << std::setw(indent) << "        Flit Size (Byte): " << noc_config.flit_size << std::endl;
            os << std::setw(indent) << "        Power (W): " << noc_config.power << std::endl;
            os << std::setw(indent) << "        Area: " << noc_config.area << std::endl;
        }
    }
};


typedef std::shared_ptr<ChipConfig> ArchConfig;


}


#endif // _ATLASIM_ARCH_H
