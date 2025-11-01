#!/bin/bash
# Comprehensive Test Script for AGNT5 (Shell version)
#
# This script provides quick curl-based testing of all 6 comprehensive scenarios.
# Useful for CI/CD pipelines and quick smoke tests.
#
# Usage:
#   ./test_comprehensive.sh              # Run all tests
#   ./test_comprehensive.sh 1            # Run specific scenario
#   ./test_comprehensive.sh --verify     # Run all with MCP verification prompts

set -e

BASE_URL="http://localhost:34181/v1/run/workflow"
TENANT_ID="${AGNT5_TENANT_ID:-00000000-0000-0000-0000-00001f718175}"
DEPLOYMENT_ID="${AGNT5_DEPLOYMENT_ID:-00000000-0000-0000-0000-00001f718175}"
COLORS=true

# Color codes
if [ "$COLORS" = true ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    CYAN='\033[0;36m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    PURPLE=''
    CYAN=''
    NC=''
fi

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    local text="$1"
    echo ""
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${CYAN}$text${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
}

print_scenario() {
    local num="$1"
    local desc="$2"
    echo -e "${PURPLE}Test Scenario $num: $desc${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_mcp_verify() {
    local run_id="$1"
    echo ""
    echo -e "${YELLOW}📊 MCP Verification for run_id: $run_id${NC}"
    echo ""
    echo "To verify via MCP, run these commands:"
    echo ""
    echo -e "${CYAN}# Check logs${NC}"
    echo "  mcp agnt5 list_run_logs --run_id=$run_id"
    echo ""
    echo -e "${CYAN}# Check traces${NC}"
    echo "  mcp agnt5 list_run_traces --run_id=$run_id"
    echo ""
    echo -e "${CYAN}# Check events${NC}"
    echo "  mcp agnt5 list_run_events --run_id=$run_id"
    echo ""
    echo -e "${CYAN}# Check entity${NC}"
    echo "  mcp agnt5 get_entity --entity_key=workflow:$run_id"
    echo ""
}

# ============================================================================
# TEST SCENARIOS
# ============================================================================

test_scenario_1() {
    print_scenario "1" "Workflow + Agent (Basic)"
    echo "Workflow: test_agent_basic"
    echo "Task: Explain what a REST API is"
    echo ""

    RESULT=$(curl -s --max-time 330 "$BASE_URL/test_agent_basic" \
        -H "Content-Type: application/json" \
        -H "X-TENANT-ID: $TENANT_ID" \
        -H "X-DEPLOYMENT-ID: $DEPLOYMENT_ID" \
        --data-raw '{"task": "Explain what a REST API is in simple terms (2-3 sentences)."}')

    RUN_ID=$(echo "$RESULT" | jq -r '.run_id')
    OUTPUT=$(echo "$RESULT" | jq -r '.output')

    if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ]; then
        print_success "Test completed - run_id: $RUN_ID"
        echo "Output: ${OUTPUT:0:100}..."

        if [ "$VERIFY_MCP" = true ]; then
            print_mcp_verify "$RUN_ID"
        fi
        return 0
    else
        print_error "Test failed"
        echo "$RESULT" | jq '.'
        return 1
    fi
}

test_scenario_2() {
    print_scenario "2" "Workflow + Agent + Simple Tool"
    echo "Workflow: test_agent_with_simple_tool"
    echo "Task: Calculate 25 * 4 + 10"
    echo ""

    RESULT=$(curl -s --max-time 330 "$BASE_URL/test_agent_with_simple_tool" \
        -H "Content-Type: application/json" \
        -H "X-TENANT-ID: $TENANT_ID" \
        -H "X-DEPLOYMENT-ID: $DEPLOYMENT_ID" \
        --data-raw '{"task": "Calculate the result of 25 * 4 + 10"}')

    RUN_ID=$(echo "$RESULT" | jq -r '.run_id')
    OUTPUT=$(echo "$RESULT" | jq -r '.output')
    TOOL_CALLS=$(echo "$RESULT" | jq -r '.tool_calls_count')

    if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ]; then
        print_success "Test completed - run_id: $RUN_ID"
        echo "Output: ${OUTPUT:0:100}..."
        echo "Tool calls: $TOOL_CALLS"

        if [ "$VERIFY_MCP" = true ]; then
            print_mcp_verify "$RUN_ID"
        fi
        return 0
    else
        print_error "Test failed"
        echo "$RESULT" | jq '.'
        return 1
    fi
}

test_scenario_3() {
    print_scenario "3" "Workflow + Agent + Multiple Tools"
    echo "Workflow: test_agent_with_multiple_tools"
    echo "Task: Search, calculate, and get weather"
    echo ""

    RESULT=$(curl -s --max-time 330 "$BASE_URL/test_agent_with_multiple_tools" \
        -H "Content-Type: application/json" \
        -H "X-TENANT-ID: $TENANT_ID" \
        -H "X-DEPLOYMENT-ID: $DEPLOYMENT_ID" \
        --data-raw '{"task": "Search for information about Python programming, calculate 15 * 8, and tell me the weather in San Francisco"}')

    RUN_ID=$(echo "$RESULT" | jq -r '.run_id')
    OUTPUT=$(echo "$RESULT" | jq -r '.output')
    TOOLS_USED=$(echo "$RESULT" | jq -r '.unique_tools_used')

    if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ]; then
        print_success "Test completed - run_id: $RUN_ID"
        echo "Output: ${OUTPUT:0:100}..."
        echo "Tools used: $TOOLS_USED"

        if [ "$VERIFY_MCP" = true ]; then
            print_mcp_verify "$RUN_ID"
        fi
        return 0
    else
        print_error "Test failed"
        echo "$RESULT" | jq '.'
        return 1
    fi
}

test_scenario_4() {
    print_scenario "4" "Workflow + Agent + Tool + Agent as Tool"
    echo "Workflow: test_agent_with_agent_as_tool"
    echo "Task: Research and analyze with specialist"
    echo ""

    RESULT=$(curl -s --max-time 330 "$BASE_URL/test_agent_with_agent_as_tool" \
        -H "Content-Type: application/json" \
        -H "X-TENANT-ID: $TENANT_ID" \
        -H "X-DEPLOYMENT-ID: $DEPLOYMENT_ID" \
        --data-raw '{"task": "Search for information about AI trends and then perform a detailed domain-specific analysis of the findings"}')

    RUN_ID=$(echo "$RESULT" | jq -r '.run_id')
    OUTPUT=$(echo "$RESULT" | jq -r '.output')
    AGENT_DELEGATIONS=$(echo "$RESULT" | jq -r '.agent_delegations')
    AGENTS=$(echo "$RESULT" | jq -r '.agents_in_workflow')

    if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ]; then
        print_success "Test completed - run_id: $RUN_ID"
        echo "Output: ${OUTPUT:0:100}..."
        echo "Agent delegations: $AGENT_DELEGATIONS"
        echo "Agents in workflow: $AGENTS"

        if [ "$VERIFY_MCP" = true ]; then
            print_mcp_verify "$RUN_ID"
        fi
        return 0
    else
        print_error "Test failed"
        echo "$RESULT" | jq '.'
        return 1
    fi
}

test_scenario_5() {
    print_scenario "5" "Workflow + Agent + HITL"
    echo "Workflow: test_agent_with_hitl"
    echo "Task: Plan a project with human input"
    echo ""

    print_info "Note: HITL scenarios may require manual interaction or auto-approval"

    RESULT=$(curl -s --max-time 330 "$BASE_URL/test_agent_with_hitl" \
        -H "Content-Type: application/json" \
        -H "X-TENANT-ID: $TENANT_ID" \
        -H "X-DEPLOYMENT-ID: $DEPLOYMENT_ID" \
        --data-raw '{"task": "Help me plan a project. Ask me what project I am working on, then suggest next steps.", "auto_approve": true}')

    RUN_ID=$(echo "$RESULT" | jq -r '.run_id')
    OUTPUT=$(echo "$RESULT" | jq -r '.output')
    HITL_INTERACTIONS=$(echo "$RESULT" | jq -r '.hitl_interactions')

    if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ]; then
        print_success "Test completed - run_id: $RUN_ID"
        echo "Output: ${OUTPUT:0:100}..."
        echo "HITL interactions: $HITL_INTERACTIONS"

        if [ "$VERIFY_MCP" = true ]; then
            print_mcp_verify "$RUN_ID"
        fi
        return 0
    else
        print_error "Test failed"
        echo "$RESULT" | jq '.'
        return 1
    fi
}

test_scenario_6() {
    print_scenario "6" "Complete Integration Test (All Features)"
    echo "Workflow: test_comprehensive_scenario"
    echo "Task: Weather, calculation, specialist, approval"
    echo ""

    RESULT=$(curl -s --max-time 330 "$BASE_URL/test_comprehensive_scenario" \
        -H "Content-Type: application/json" \
        -H "X-TENANT-ID: $TENANT_ID" \
        -H "X-DEPLOYMENT-ID: $DEPLOYMENT_ID" \
        --data-raw '{"task": "Research current weather in New York, calculate travel time if driving 60 mph for 3.5 hours, delegate detailed analysis to a specialist, and request approval for the final plan."}')

    RUN_ID=$(echo "$RESULT" | jq -r '.run_id')
    OUTPUT=$(echo "$RESULT" | jq -r '.output')
    TOTAL_TOOLS=$(echo "$RESULT" | jq -r '.total_tool_calls')
    AGENT_DELEGATIONS=$(echo "$RESULT" | jq -r '.agent_delegations')
    HITL=$(echo "$RESULT" | jq -r '.hitl_interactions')

    if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "null" ]; then
        print_success "Test completed - run_id: $RUN_ID"
        echo "Output: ${OUTPUT:0:100}..."
        echo "Total tool calls: $TOTAL_TOOLS"
        echo "Agent delegations: $AGENT_DELEGATIONS"
        echo "HITL interactions: $HITL"

        if [ "$VERIFY_MCP" = true ]; then
            print_mcp_verify "$RUN_ID"
        fi
        return 0
    else
        print_error "Test failed"
        echo "$RESULT" | jq '.'
        return 1
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    local scenario="${1:-all}"
    VERIFY_MCP=false

    if [ "$scenario" = "--verify" ]; then
        VERIFY_MCP=true
        scenario="all"
    fi

    print_header "COMPREHENSIVE AGNT5 TESTING"

    if [ "$scenario" = "all" ]; then
        echo "Running all 6 test scenarios..."
        echo ""

        PASSED=0
        FAILED=0

        test_scenario_1 && ((PASSED++)) || ((FAILED++))
        echo ""
        test_scenario_2 && ((PASSED++)) || ((FAILED++))
        echo ""
        test_scenario_3 && ((PASSED++)) || ((FAILED++))
        echo ""
        test_scenario_4 && ((PASSED++)) || ((FAILED++))
        echo ""
        test_scenario_5 && ((PASSED++)) || ((FAILED++))
        echo ""
        test_scenario_6 && ((PASSED++)) || ((FAILED++))

        # Summary
        print_header "TEST SUMMARY"
        echo "Total tests: 6"
        echo -e "${GREEN}Passed: $PASSED${NC}"
        echo -e "${RED}Failed: $FAILED${NC}"
        echo ""

        if [ $FAILED -eq 0 ]; then
            print_success "ALL TESTS PASSED"
            exit 0
        else
            print_error "SOME TESTS FAILED"
            exit 1
        fi

    else
        # Run specific scenario
        case "$scenario" in
            1) test_scenario_1 ;;
            2) test_scenario_2 ;;
            3) test_scenario_3 ;;
            4) test_scenario_4 ;;
            5) test_scenario_5 ;;
            6) test_scenario_6 ;;
            *)
                echo "Invalid scenario: $scenario"
                echo "Usage: $0 [1-6|all|--verify]"
                exit 1
                ;;
        esac
    fi
}

# Run main
main "$@"
