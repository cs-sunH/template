/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Persistent 3D-ICE 4.0 bridge for wafercosim.
 *
 * This program links to 3D-ICE and its bundled SuperLU-MT solver. It keeps one
 * ThermalData_t alive for the process lifetime and advances it with
 * emulate_step(). Project block powers are mapped to the active source layer
 * by cell/block area overlap; reported block temperatures are area-weighted
 * averages over the same overlaps. Coordinates in the manifest are micrometres.
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "analysis.h"
#include "dimensions.h"
#include "output.h"
#include "power_grid.h"
#include "stack_description.h"
#include "stack_file_parser.h"
#include "thermal_data.h"
#include "types.h"

typedef struct {
    double x;
    double y;
    double width;
    double height;
    double area;
} BlockRect;

typedef struct {
    CellIndex_t cell;
    double overlap_area;
} CellOverlap;

typedef struct {
    CellOverlap *items;
    size_t count;
    size_t capacity;
} BlockMap;

static void fail(const char *message)
{
    fprintf(stderr, "ice_bridge: %s\n", message);
}

static int append_overlap(BlockMap *map, CellIndex_t cell, double area)
{
    if (map->count == map->capacity) {
        size_t next = map->capacity == 0 ? 16 : map->capacity * 2;
        CellOverlap *grown = (CellOverlap *)realloc(map->items, next * sizeof(*grown));
        if (grown == NULL) return 0;
        map->items = grown;
        map->capacity = next;
    }
    map->items[map->count].cell = cell;
    map->items[map->count].overlap_area = area;
    map->count++;
    return 1;
}

static double overlap_1d(double a0, double a1, double b0, double b1)
{
    double lo = a0 > b0 ? a0 : b0;
    double hi = a1 < b1 ? a1 : b1;
    return hi > lo ? hi - lo : 0.0;
}

static int is_source_layer(StackLayerType_t type)
{
    return type == TDICE_LAYER_SOURCE ||
           type == TDICE_LAYER_SOURCE_CONNECTED_TO_AMBIENT ||
           type == TDICE_LAYER_SOURCE_CONNECTED_TO_SPREADER ||
           type == TDICE_LAYER_SOURCE_CONNECTED_TO_PCB;
}

static int read_manifest(const char *path, BlockRect **blocks_out, size_t *count_out)
{
    FILE *file = fopen(path, "r");
    char magic[32];
    unsigned version = 0;
    unsigned count = 0;
    if (file == NULL) {
        fprintf(stderr, "ice_bridge: cannot open block manifest %s: %s\n", path, strerror(errno));
        return 0;
    }
    if (fscanf(file, "%31s %u %u", magic, &version, &count) != 3 ||
        strcmp(magic, "ICEBLOCKS") != 0 || version != 1 || count == 0) {
        fail("invalid manifest header; expected 'ICEBLOCKS 1 <positive-block-count>'");
        fclose(file);
        return 0;
    }
    BlockRect *blocks = (BlockRect *)calloc(count, sizeof(*blocks));
    if (blocks == NULL) {
        fail("could not allocate block manifest");
        fclose(file);
        return 0;
    }
    for (unsigned i = 0; i < count; ++i) {
        BlockRect *b = &blocks[i];
        if (fscanf(file, "%lf %lf %lf %lf", &b->x, &b->y, &b->width, &b->height) != 4 ||
            !isfinite(b->x) || !isfinite(b->y) || !isfinite(b->width) ||
            !isfinite(b->height) || b->x < 0.0 || b->y < 0.0 ||
            b->width <= 0.0 || b->height <= 0.0) {
            fprintf(stderr, "ice_bridge: invalid rectangle for block %u\n", i);
            free(blocks);
            fclose(file);
            return 0;
        }
        b->area = b->width * b->height;
    }
    if (fclose(file) != 0) {
        fail("failed to close block manifest");
        free(blocks);
        return 0;
    }
    *blocks_out = blocks;
    *count_out = count;
    return 1;
}

static int build_area_maps(
    Dimensions_t *dimensions,
    CellIndex_t source_layer,
    const BlockRect *blocks,
    size_t nblocks,
    BlockMap **maps_out)
{
    if (dimensions->NonUniform != 0) {
        fail("only regular uniform-y 4RM grids are supported by the project-block mapper");
        return 0;
    }
    CellIndex_t rows = dimensions->Grid.NRows;
    CellIndex_t columns = dimensions->Grid.NColumns;
    if (rows == 0 || columns == 0 || source_layer >= dimensions->Grid.NLayers) {
        fail("invalid 3D-ICE grid dimensions or source-layer index");
        return 0;
    }
    BlockMap *maps = (BlockMap *)calloc(nblocks, sizeof(*maps));
    if (maps == NULL) {
        fail("could not allocate area maps");
        return 0;
    }

    for (size_t bidx = 0; bidx < nblocks; ++bidx) {
        const BlockRect *b = &blocks[bidx];
        double covered = 0.0;
        for (CellIndex_t row = 0; row < rows; ++row) {
            double cy0 = get_cell_location_y(dimensions, row);
            double cy1 = cy0 + get_cell_width(dimensions, row);
            double dy = overlap_1d(b->y, b->y + b->height, cy0, cy1);
            if (dy == 0.0) continue;
            for (CellIndex_t col = 0; col < columns; ++col) {
                double cx0 = get_cell_location_x(dimensions, col);
                double cx1 = cx0 + get_cell_length(dimensions, col);
                double dx = overlap_1d(b->x, b->x + b->width, cx0, cx1);
                if (dx == 0.0) continue;
                double area = dx * dy;
                CellIndex_t cell = get_cell_offset_in_stack(dimensions, source_layer, row, col);
                if (!append_overlap(&maps[bidx], cell, area)) {
                    fail("could not grow area-overlap map");
                    for (size_t i = 0; i < nblocks; ++i) free(maps[i].items);
                    free(maps);
                    return 0;
                }
                covered += area;
            }
        }
        double tolerance = fmax(1e-6, b->area * 1e-10);
        if (maps[bidx].count == 0 || fabs(covered - b->area) > tolerance) {
            fprintf(stderr,
                    "ice_bridge: block %zu is not fully covered by the 3D-ICE grid "
                    "(block=%.12g um^2, overlap=%.12g um^2)\n",
                    bidx, b->area, covered);
            for (size_t i = 0; i < nblocks; ++i) free(maps[i].items);
            free(maps);
            return 0;
        }
    }
    *maps_out = maps;
    return 1;
}

static char *next_token(char **cursor)
{
    char *p = *cursor;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ',') p++;
    if (*p == '\0') {
        *cursor = p;
        return NULL;
    }
    char *start = p;
    while (*p != '\0' && *p != ',' && *p != '\r' && *p != '\n') p++;
    if (*p != '\0') *p++ = '\0';
    *cursor = p;
    while (*start == ' ' || *start == '\t') start++;
    char *end = start + strlen(start);
    while (end > start && (end[-1] == ' ' || end[-1] == '\t')) *--end = '\0';
    return start;
}

static int parse_double_token(char *token, double *out)
{
    char *end = NULL;
    errno = 0;
    double value = strtod(token, &end);
    if (errno != 0 || end == token || *end != '\0' || !isfinite(value)) return 0;
    *out = value;
    return 1;
}

static void cleanup(
    int thermal_ready,
    int parsed,
    ThermalData_t *thermal,
    StackDescription_t *stack,
    Analysis_t *analysis,
    Output_t *output)
{
    if (thermal_ready) thermal_data_destroy(thermal);
    if (parsed) {
        output_destroy(output);
        analysis_destroy(analysis);
        stack_description_destroy(stack);
    }
}

int main(int argc, char **argv)
{
    /* stdout is a pipe under the Python parent; force line buffering before
       thermal_data_build performs any stdio on this stream. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc != 5 || strcmp(argv[1], "--stack") != 0 || strcmp(argv[3], "--blocks") != 0) {
        fprintf(stderr, "Usage: %s --stack stack.stk --blocks project-blocks.map\n", argv[0]);
        return EXIT_FAILURE;
    }

    BlockRect *blocks = NULL;
    size_t nblocks = 0;
    BlockMap *maps = NULL;
    StackDescription_t stack;
    Analysis_t analysis;
    Output_t output;
    ThermalData_t thermal;
    Source_t *baseline_sources = NULL;
    int parsed = 0;
    int thermal_ready = 0;
    int rc = EXIT_FAILURE;
    int saved_stdout = -1;

    if (!read_manifest(argv[4], &blocks, &nblocks)) goto done;

    stack_description_init(&stack);
    analysis_init(&analysis);
    output_init(&output);
    thermal_data_init(&thermal);
    parsed = 1;

    /* 3D-ICE writes solver setup timings to stdout. Keep the bridge protocol
       machine-readable by routing initialization chatter to stderr. */
    fflush(stdout);
    saved_stdout = dup(STDOUT_FILENO);
    if (saved_stdout < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        fail("could not redirect initialization diagnostics");
        goto done;
    }

    if (parse_stack_description_file((String_t)argv[2], &stack, &analysis, &output) != TDICE_SUCCESS) {
        fail("3D-ICE could not parse the stack file");
        goto restore_stdout;
    }
    if (analysis.AnalysisType != TDICE_ANALYSIS_TYPE_TRANSIENT || analysis.StepTime <= 0.0) {
        fail("stack must configure a positive transient step");
        goto restore_stdout;
    }
    if (stack.Channel == NULL || stack.Channel->ChannelModel != TDICE_CHANNEL_MODEL_MC_4RM) {
        fail("this bridge requires one regular 4RM microchannel stack");
        goto restore_stdout;
    }
    if (thermal_data_build(&thermal, &stack.StackElements, stack.Dimensions, &analysis, &stack.Materials) != TDICE_SUCCESS) {
        fail("3D-ICE thermal_data_build failed");
        goto restore_stdout;
    }
    thermal_ready = 1;
    if (update_source_vector(&thermal.PowerGrid, stack.Dimensions) != TDICE_SUCCESS) {
        fail("3D-ICE could not initialize floorplan/coolant baseline source terms");
        goto restore_stdout;
    }
    baseline_sources = (Source_t *)malloc(thermal.PowerGrid.NCells * sizeof(*baseline_sources));
    if (baseline_sources == NULL) {
        fail("could not preserve 3D-ICE baseline sources");
        goto restore_stdout;
    }
    memcpy(baseline_sources, thermal.PowerGrid.Sources,
           thermal.PowerGrid.NCells * sizeof(*baseline_sources));

restore_stdout:
    fflush(stdout);
    if (saved_stdout >= 0) {
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);
        saved_stdout = -1;
    }
    if (rc == EXIT_FAILURE && analysis.AnalysisType != TDICE_ANALYSIS_TYPE_TRANSIENT) goto done;
    if (thermal_ready == 0 || baseline_sources == NULL) goto done;

    Dimensions_t *dimensions = stack.Dimensions;
    if (dimensions == NULL || dimensions->NonUniform != 0) {
        fail("stack must use a regular uniform-y grid; non-uniform grids are not mapped");
        goto done;
    }
    if (thermal.PowerGrid.NCells != dimensions->Grid.NCells ||
        thermal.Size != dimensions->Grid.NCells) {
        fail("unexpected 3D-ICE power/temperature grid layout");
        goto done;
    }

    CellIndex_t source_layer = UINT32_MAX;
    unsigned source_layers = 0;
    for (CellIndex_t layer = 0; layer < thermal.PowerGrid.NLayers; ++layer) {
        if (is_source_layer(thermal.PowerGrid.LayersTypeProfile[layer])) {
            source_layer = layer;
            source_layers++;
        }
    }
    if (source_layers != 1) {
        fprintf(stderr, "ice_bridge: expected one active source layer, found %u\n", source_layers);
        goto done;
    }
    if (!build_area_maps(dimensions, source_layer, blocks, nblocks, &maps)) goto done;

    /* The factorization in 3D-ICE is built for Analysis.StepTime. Keep the
       native state and matrix untouched; use a long synthetic slot so that
       emulate_step does not replace our per-step Sources from a batch queue. */
    analysis.CurrentTime = 1u;
    analysis.SlotLength = UINT32_MAX;

    printf("# time_s");
    for (size_t i = 0; i < nblocks; ++i) printf(",b%zu", i);
    printf("\n");

    char *line = NULL;
    size_t line_capacity = 0;
    double elapsed_s = 0.0;
    size_t row_number = 0;
    for (;;) {
        ssize_t length = getline(&line, &line_capacity, stdin);
        if (length < 0) {
            if (feof(stdin)) {
                rc = EXIT_SUCCESS;
                break;
            }
            fail("failed reading a power row");
            break;
        }
        char *cursor = line;
        char *token = next_token(&cursor);
        if (token == NULL) continue;
        double dt = 0.0;
        if (!parse_double_token(token, &dt) || dt <= 0.0 ||
            fabs(dt - analysis.StepTime) > fmax(1e-12, analysis.StepTime * 1e-9)) {
            fprintf(stderr,
                    "ice_bridge: row %zu dt must match fixed stack step %.12g s\n",
                    row_number + 1, analysis.StepTime);
            break;
        }

        double *power = (double *)calloc(nblocks, sizeof(*power));
        double *cell_power = (double *)calloc(thermal.PowerGrid.NCells, sizeof(*cell_power));
        if (power == NULL || cell_power == NULL) {
            free(power);
            free(cell_power);
            fail("could not allocate per-step power arrays");
            break;
        }
        double total_input_w = 0.0;
        int row_ok = 1;
        for (size_t i = 0; i < nblocks; ++i) {
            token = next_token(&cursor);
            if (token == NULL || !parse_double_token(token, &power[i]) || power[i] < 0.0) {
                fprintf(stderr, "ice_bridge: invalid power value at row %zu block %zu\n", row_number + 1, i);
                row_ok = 0;
                break;
            }
            total_input_w += power[i];
        }
        if (row_ok && next_token(&cursor) != NULL) {
            fprintf(stderr, "ice_bridge: too many values in power row %zu\n", row_number + 1);
            row_ok = 0;
        }
        if (!row_ok) {
            free(power);
            free(cell_power);
            break;
        }

        for (size_t b = 0; b < nblocks; ++b) {
            for (size_t j = 0; j < maps[b].count; ++j) {
                CellOverlap entry = maps[b].items[j];
                cell_power[entry.cell] += power[b] * entry.overlap_area / blocks[b].area;
            }
        }
        double total_mapped_w = 0.0;
        memcpy(thermal.PowerGrid.Sources, baseline_sources,
               thermal.PowerGrid.NCells * sizeof(*baseline_sources));
        for (CellIndex_t i = 0; i < thermal.PowerGrid.NCells; ++i) {
            thermal.PowerGrid.Sources[i] += cell_power[i];
            total_mapped_w += cell_power[i];
        }
        if (fabs(total_mapped_w - total_input_w) > fmax(1e-12, fabs(total_input_w) * 1e-10)) {
            fail("area-overlap power mapping did not conserve total watts");
            free(power);
            free(cell_power);
            break;
        }

        SimResult_t result = emulate_step(&thermal, dimensions, &analysis);
        if (result != TDICE_STEP_DONE && result != TDICE_SLOT_DONE) {
            fprintf(stderr, "ice_bridge: emulate_step failed with code %d at row %zu\n", result, row_number + 1);
            free(power);
            free(cell_power);
            break;
        }

        elapsed_s += dt;
        printf("%.12g", elapsed_s);
        for (size_t b = 0; b < nblocks; ++b) {
            double weighted_temperature = 0.0;
            for (size_t j = 0; j < maps[b].count; ++j) {
                CellOverlap entry = maps[b].items[j];
                weighted_temperature += thermal.Temperatures[entry.cell] * entry.overlap_area;
            }
            double block_temperature = weighted_temperature / blocks[b].area;
            if (!isfinite(block_temperature)) {
                fail("non-finite mapped block temperature");
                row_ok = 0;
                break;
            }
            printf(",%.12g", block_temperature);
        }
        printf("\n");
        fflush(stdout);
        free(power);
        free(cell_power);
        if (!row_ok) break;
        row_number++;
    }
    free(line);

done:
    if (saved_stdout >= 0) {
        fflush(stdout);
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);
    }
    if (maps != NULL) {
        for (size_t i = 0; i < nblocks; ++i) free(maps[i].items);
        free(maps);
    }
    free(baseline_sources);
    free(blocks);
    cleanup(thermal_ready, parsed, &thermal, &stack, &analysis, &output);
    return rc;
}
