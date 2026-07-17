#include "geo_identity_corpus.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

template <int ID>
bool check_identity(uint64_t assignments) {
    using identity_t = geo_identity_generated::identity<ID>;
    bool found_counterexample = false;

    for (uint64_t assignment = 0; assignment < assignments; ++assignment) {
        const auto witness = identity_t::evaluate(assignment);
        if (!witness.equal) {
            found_counterexample = true;
            if (!identity_t::EXPECT_COUNTEREXAMPLE) {
                std::fprintf(
                    stderr,
                    "HOST_SMOKE_FAIL,name=%s,assignment=%llu,blade=%u,lhs=%d,rhs=%d\n",
                    identity_t::NAME,
                    static_cast<unsigned long long>(assignment),
                    static_cast<unsigned int>(witness.blade),
                    witness.lhs,
                    witness.rhs
                );
                return false;
            }
            std::printf(
                "HOST_COUNTEREXAMPLE,name=%s,assignment=%llu,blade=%u,lhs=%d,rhs=%d\n",
                identity_t::NAME,
                static_cast<unsigned long long>(assignment),
                static_cast<unsigned int>(witness.blade),
                witness.lhs,
                witness.rhs
            );
            break;
        }
    }

    if (identity_t::EXPECT_COUNTEREXAMPLE && !found_counterexample) {
        std::fprintf(
            stderr,
            "HOST_SMOKE_FAIL,name=%s,reason=no_counterexample\n",
            identity_t::NAME
        );
        return false;
    }

    std::printf(
        "HOST_IDENTITY_RESULT,name=%s,expected=%s,status=pass\n",
        identity_t::NAME,
        identity_t::EXPECT_COUNTEREXAMPLE ? "counterexample" : "identity"
    );
    return true;
}

bool dispatch(int identity_index, uint64_t assignments) {
    switch (identity_index) {
#define GEO_HOST_CASE(ID) \
        case ID: return check_identity<ID>(assignments);
        GEO_IDENTITY_FOR_EACH(GEO_HOST_CASE)
#undef GEO_HOST_CASE
        default:
            return false;
    }
}

}  // namespace

int main() {
    constexpr uint64_t assignments = UINT64_C(4096);
    bool passed = true;
    for (
        int identity_index = 0;
        identity_index < geo_identity_generated::IDENTITY_COUNT;
        ++identity_index
    ) {
        passed = dispatch(identity_index, assignments) && passed;
    }
    std::printf(
        "GEO_IDENTITY_HOST_SMOKE,status=%s,identities=%d,assignments=%llu\n",
        passed ? "complete" : "fail",
        geo_identity_generated::IDENTITY_COUNT,
        static_cast<unsigned long long>(assignments)
    );
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
