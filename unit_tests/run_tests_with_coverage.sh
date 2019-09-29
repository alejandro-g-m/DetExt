branch=false
missing=false
all=false

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

# Shows (or not) branch coverage (conditional posibilities)
if [ "$branch" = true ] ; then
    coverage run --branch unit_tests.py
else
    coverage run unit_tests.py
fi

# Shows final report for all files or only the main code ones
if [ "$all" = true ] ; then
    files="../feature_vector_creation.py ../dns_attacks_detection.py unit_tests.py unit_tests_feature_vector_creation.py unit_tests_dns_attacks_detection.py"
else
    files="../feature_vector_creation.py ../dns_attacks_detection.py"
fi

# Shows lines (or not) that have not been run in final report
if [ "$missing" = true ] ; then
    coverage report -m $files
else
    coverage report $files
fi
