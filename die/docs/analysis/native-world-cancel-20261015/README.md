# Frozen native coarse cancellation events

This change adds an opt-in `coarse_cancel_events` schedule to the active
`astra-sim-joint` native projection. The schedule is immutable input consumed
by the existing C++ event queue and `NativeCoarseDieDomain.cancel`; Python
continues to validate and serialize the wire only. The portable build and test
entry point is [`run_checks.sh`](run_checks.sh). Captured command output and
source/library identities are in
[`die/docs/evidence/native-world-cancel-20261015/`](../../evidence/native-world-cancel-20261015/).

When the schedule is absent or empty, its wire key and all cancellation-only
inspect/snapshot fields are omitted. The legacy event priorities and behavior
are unchanged. With an explicit schedule, same-tick native terminal/credit
settlement wins first, then cancellation, then ready/arrival/capacity work.
The first cancellation of a pending leg records an outer cancelled terminal
without creating a Domain binding. Cancellation of an issued coarse Plan or
endpoint goes through the canonical Domain `cancel`, preserving service
already completed and the cancelled Domain terminal. Later duplicate events
are idempotent and remain represented in the processed-prefix accounting.

Cancellation does not publish success, decrement successful work accounting,
signal a barrier, or release dependencies. A dependent of a cancelled leg
stays blocked. A target with an acquire/release lease or barrier signal is
rejected: this projection cannot prove unique acquired-lease ownership, so it
does not guess which gate units to release. The supported scope is one frozen
native coarse Plan or endpoint routed through exactly one frozen Domain, with
no gate or barrier effects. The active public kernel value object remains
unchanged. `sh30` continues to guard production coarse selection and native
inspect continues to report `live_world_import_supported=false`; this is
native local-world evidence, not a claim that production Online imports this
snapshot.

Restore checks both directions of the frozen prefix. Every consumed event's
first target event must correspond to natural completion at or before that
tick, or to a cancelled terminal exactly at that tick. A consumed prefix with
the target still active/pending, or a success terminal after the first cancel,
is rejected atomically. Tests also cover an unconsumed duplicate, source
continuation against the imported fork, same-tick terminal-wins behavior, and
an independent direct `NativeCoarseDieDomain` replay retaining a strictly
partial actual-byte prefix.

## Reproduce

Run from any directory in the Git checkout:

```sh
bash die/docs/analysis/native-world-cancel-20261015/run_checks.sh
```

The script configures the existing active aggregate under
`die/docs/analysis/native-local-projection-20261014/online-reproduction/aggregate`,
builds the actual native projection, C APIs, Domain test, and event-window test,
then runs four CTest targets and the Python-to-native/real-C-ABI suite. Override
`BUILD_DIR`, `CMAKE_BIN`, `CTEST_BIN`, `BUILD_JOBS`, or `PROTOBUF_PREFIX` when needed. The
protobuf prefix is optional if the host provides matching development headers,
libraries, and `protoc`.

Required tools are CMake 3.22 or newer, a C++14 compiler, Python 3, `protoc`,
and matching protobuf development headers/libraries. This run used real Debian
packages `libprotobuf-dev`, `protobuf-compiler`, and `libprotoc32t64`, all
version `3.21.12-11+deb13u1`, extracted without root into
`work/protobuf-root`. The machine already had matching `libprotobuf32t64`.
If system packages are unavailable, download matching packages from the
configured Debian mirror and unpack them without installing:

```sh
mkdir -p work/protobuf-debs work/protobuf-root
(
  cd work/protobuf-debs
  apt-get download \
    libprotobuf-dev=3.21.12-11+deb13u1 \
    protobuf-compiler=3.21.12-11+deb13u1 \
    libprotoc32t64=3.21.12-11+deb13u1
)
for package in work/protobuf-debs/*.deb; do
  dpkg-deb -x "$package" work/protobuf-root
done
```

Set `PROTOBUF_PREFIX` to that extracted prefix when replaying. If package versions
in the configured repository differ, use one matching the installed protobuf
runtime and update the pinned CMake paths accordingly. No protobuf stub or
mock library is used.

## Captured validation

The final current-source run passed these four native tests with no skips:
`JointPipelineWindowTest`, `NativeProjectionCApiTest`,
`NativeCoarseProjectionTest`, and `NativeCoarseDieDomainTest` (4/4). The
`joint.test_native_coarse_projection` Python suite passed all six tests with
the real `libjoint_projection.so` and `libcoarse_die_cabi.so` selected (6/6,
no environment skips). Raw logs, the read-only CTest discovery, exact commands,
protobuf package checksums, and source/library SHA-256 identities are in the
evidence directory.

The first restore-test failure was not redirected to a file at the time. Its
command and observed assertion are preserved as a clearly labeled
transcription, not represented as a raw terminal log. The cause was a mutation
callback that could miss the retained endpoint binding; the test now requires
that the binding was changed. The later reverse-prefix review added explicit
pending and post-cancel-success corruption checks.
