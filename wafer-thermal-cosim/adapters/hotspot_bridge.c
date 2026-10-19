#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "flp.h"
#include "microchannel.h"
#include "materials.h"
#include "temperature.h"
#include "temperature_block.h"
#include "temperature_grid.h"

static void usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s [--flp FLOORPLAN] [--config CONFIG] [--materials MATERIALS]\n"
        "          [--microchannels] [--detailed-3d]\n"
        "Reads one row per step from stdin: dt_seconds, W_for_each_floorplan_block.\n"
        "Writes cumulative_time_s and each block's temperature in kelvin as CSV.\n",
        program);
}

static double parse_number(char **cursor, const char *line, int line_no)
{
    char *end;
    double value;

    while (**cursor && (isspace((unsigned char)**cursor) || **cursor == ','))
        ++*cursor;
    if (!**cursor || **cursor == '#') {
        fprintf(stderr, "input line %d: missing numeric field in: %s", line_no, line);
        exit(2);
    }

    errno = 0;
    value = strtod(*cursor, &end);
    if (end == *cursor || errno == ERANGE || !isfinite(value)) {
        fprintf(stderr, "input line %d: invalid finite number in: %s", line_no, line);
        exit(2);
    }
    *cursor = end;
    return value;
}

static void check_line_end(char *cursor, const char *line, int line_no)
{
    while (*cursor && isspace((unsigned char)*cursor))
        ++cursor;
    if (*cursor && *cursor != '#') {
        fprintf(stderr, "input line %d: too many fields in: %s", line_no, line);
        exit(2);
    }
}

static void compute_temp_quiet(RC_model_t *model, double *power, double *temp,
                               double dt)
{
    int saved_stdout = dup(STDOUT_FILENO);

    fflush(stdout);
    if (saved_stdout >= 0)
        dup2(STDERR_FILENO, STDOUT_FILENO);
    compute_temp(model, power, temp, dt);
    fflush(stdout);
    if (saved_stdout >= 0) {
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);
    }
}

int main(int argc, char **argv)
{
    const char *flp_path = NULL;
    const char *config_path = NULL;
    const char *materials_path = NULL;
    int use_microchannels = 0;
    int detailed_3d = 0;
    str_pair options[MAX_ENTRIES];
    int option_count = 0;
    int i;
    int line_no = 0;
    size_t line_capacity = 0;
    char *line = NULL;
    double elapsed = 0.0;
    flp_t *flp = NULL;
    RC_model_t *model = NULL;
    double *power = NULL;
    double *temp = NULL;
    materials_list_t materials;
    microchannel_config_t *microchannel = NULL;
    thermal_config_t config;

    materials.size = 0;
    materials.names = NULL;
    materials.material_properties = NULL;
    default_materials(&materials);

    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        }
        if (!strcmp(argv[i], "--microchannels")) {
            use_microchannels = 1;
            continue;
        }
        if (!strcmp(argv[i], "--detailed-3d")) {
            detailed_3d = 1;
            continue;
        }
        if ((!strcmp(argv[i], "--flp") || !strcmp(argv[i], "--config") ||
             !strcmp(argv[i], "--materials")) && i + 1 < argc) {
            const char *flag = argv[i++];
            if (!strcmp(flag, "--flp")) flp_path = argv[i];
            else if (!strcmp(flag, "--config")) config_path = argv[i];
            else materials_path = argv[i];
            continue;
        }
        usage(argv[0]);
        return 2;
    }
    if (materials_path) {
        char materials_file[STR_SIZE];
        if (strlen(materials_path) >= sizeof(materials_file)) {
            fprintf(stderr, "materials path is too long\n");
            return 2;
        }
        strcpy(materials_file, materials_path);
        materials_add_from_file(&materials, materials_file);
    }
    if (config_path) {
        char config_file[STR_SIZE];
        if (strlen(config_path) >= sizeof(config_file)) {
            fprintf(stderr, "config path is too long\n");
            free_materials(&materials);
            return 2;
        }
        strcpy(config_file, config_path);
        option_count = read_str_pairs(options, MAX_ENTRIES, config_file);
    }

    config = default_thermal_config();
    thermal_config_add_from_strs(&config, &materials, options, option_count);
    if (strcmp(config.model_type, BLOCK_MODEL_STR) && strcmp(config.model_type, GRID_MODEL_STR)) {
        fprintf(stderr, "unsupported HotSpot model type: %s\n", config.model_type);
        free_materials(&materials);
        return 2;
    }
    if (use_microchannels && (strcmp(config.model_type, GRID_MODEL_STR) || !detailed_3d)) {
        fprintf(stderr, "--microchannels requires grid mode and --detailed-3d\n");
        free_materials(&materials);
        return 2;
    }
    if (detailed_3d && strcmp(config.model_type, GRID_MODEL_STR)) {
        fprintf(stderr, "--detailed-3d requires grid mode\n");
        free_materials(&materials);
        return 2;
    }

    if (!strcmp(config.grid_layer_file, NULLFILE)) {
        char floorplan_file[STR_SIZE];
        if (!flp_path) {
            fprintf(stderr, "--flp is required when grid_layer_file is not set\n");
            free_materials(&materials);
            return 2;
        }
        if (strlen(flp_path) >= sizeof(floorplan_file)) {
            fprintf(stderr, "floorplan path is too long\n");
            free_materials(&materials);
            return 2;
        }
        strcpy(floorplan_file, flp_path);
        flp = read_flp(floorplan_file, FALSE, FALSE);
    }
    if (use_microchannels) {
        if (strcmp(config.grid_layer_file, NULLFILE) == 0) {
            fprintf(stderr, "--microchannels requires a layer configuration file\n");
            if (flp) free_flp(flp, FALSE, FALSE);
            free_materials(&materials);
            return 2;
        }
        microchannel = malloc(sizeof(*microchannel));
        if (!microchannel) {
            fprintf(stderr, "could not allocate microchannel configuration\n");
            if (flp) free_flp(flp, FALSE, FALSE);
            free_materials(&materials);
            return 2;
        }
        *microchannel = default_microchannel_config();
        microchannel_config_add_from_strs(microchannel, &materials, options, option_count);
    }

    {
        int saved_stdout = dup(STDOUT_FILENO);
        fflush(stdout);
        if (saved_stdout >= 0)
            dup2(STDERR_FILENO, STDOUT_FILENO);
        model = alloc_RC_model(&config, flp, microchannel, &materials, detailed_3d, use_microchannels);
        populate_R_model(model, flp);
        populate_C_model(model, flp);
        fflush(stdout);
        if (saved_stdout >= 0) {
            dup2(saved_stdout, STDOUT_FILENO);
            close(saved_stdout);
        }
    }
    power = hotspot_vector(model);
    temp = hotspot_vector(model);
    if (strcmp(model->config->init_file, NULLFILE))
        read_temp(model, temp, model->config->init_file, FALSE);
    else
        set_temp(model, temp, model->config->init_temp);

    {
        int n_power_blocks;
        if (model->type == BLOCK_MODEL) {
            n_power_blocks = model->block->n_units;
            printf("# time_s");
            for (i = 0; i < n_power_blocks; ++i)
                printf(",%s_K", model->block->flp->units[i].name);
        } else {
            int layer;
            n_power_blocks = model->grid->total_n_blocks;
            printf("# time_s");
            for (layer = 0; layer < model->grid->n_layers; ++layer) {
                int block;
                for (block = 0; block < model->grid->layers[layer].flp->n_units; ++block) {
                    printf(",L%d_%s_K", model->grid->layers[layer].no,
                           model->grid->layers[layer].flp->units[block].name);
                }
            }
        }
        putchar('\n');
        fflush(stdout);
    }

    while (getline(&line, &line_capacity, stdin) >= 0) {
        char *cursor = line;
        double dt;
        int n_power_blocks;
        int state_nodes;
        static int first_grid_step = 1;

        ++line_no;
        while (*cursor && isspace((unsigned char)*cursor))
            ++cursor;
        if (!*cursor || *cursor == '#')
            continue;

        dt = parse_number(&cursor, line, line_no);
        if (dt <= 0.0) {
            fprintf(stderr, "input line %d: dt_seconds must be positive\n", line_no);
            return 2;
        }
        if (model->type == BLOCK_MODEL) {
            n_power_blocks = model->block->n_units;
            state_nodes = model->block->n_nodes;
        } else {
            n_power_blocks = model->grid->total_n_blocks;
            state_nodes = n_power_blocks + (model->grid->config.model_secondary ? EXTRA + EXTRA_SEC : EXTRA);
        }
        zero_dvector(power, state_nodes);
        for (i = 0; i < n_power_blocks; ++i) {
            power[i] = parse_number(&cursor, line, line_no);
            if (power[i] < 0.0) {
                fprintf(stderr, "input line %d: block power must be nonnegative\n", line_no);
                return 2;
            }
        }
        check_line_end(cursor, line, line_no);

        /* The same RC model and temperature vector survive every input row. */
        if (model->type == GRID_MODEL && !first_grid_step)
            compute_temp_quiet(model, power, NULL, dt);
        else
            compute_temp_quiet(model, power, temp, dt);
        first_grid_step = 0;
        elapsed += dt;
        printf("%.12g", elapsed);
        for (i = 0; i < n_power_blocks; ++i)
            printf(",%.12g", temp[i]);
        putchar('\n');
        fflush(stdout);
    }

    free(line);
    free_dvector(power);
    free_dvector(temp);
    delete_RC_model(model);
    if (flp) free_flp(flp, FALSE, FALSE);
    free_materials(&materials);
    free_microchannel(microchannel);
    return 0;
}
