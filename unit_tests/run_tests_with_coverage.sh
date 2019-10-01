branch=false
missing=false
all=false
core_files="../feature_vector_creation.py,../dns_attacks_detection.py,../parse_BRO_log_file.py,../live_sniffer.py"
test_files="unit_tests.py,unit_tests_feature_vector_creation.py,unit_tests_dns_attacks_detection.py"


while [ "$1" != "" ]; do
    case $1 in
        -b | --branch )          branch=true
                                 ;;
        -m | --show-missing )    missing=true
                                 ;;
        -a | --all )             all=true
    esac
    shift
done

# Shows branch coverage (conditional posibilities)
if [ "$branch" = true ] ; then
    coverage run --include="$core_files,$test_files" --branch unit_tests.py
else
    coverage run --include="$core_files,$test_files" unit_tests.py
fi

# Shows final report for all files or only the main code ones
if [ "$all" = true ] ; then
    report_files="$core_files,$test_files"
else
    report_files="$core_files"
fi

# Shows lines that have not been run in final report
if [ "$missing" = true ] ; then
    coverage report -m --include="$report_files"
else
    coverage report --include="$report_files"
fi

# Generate HTML report
coverage html --title="Test coverage DetExt" --include="$report_files" -d  coverage_report
