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

if [ "$branch" = true ] ; then
    coverage run --branch unit_tests.py
else
    coverage run unit_tests.py
fi

if [ "$all" = true ] ; then
    files="feature_vector_creation.py dns_attacks_detection.py unit_tests.py unit_tests_feature_vector_creation.py unit_tests_dns_attacks_detection.py"
else
    files="feature_vector_creation.py dns_attacks_detection.py"
fi

if [ "$missing" = true ] ; then
    coverage report -m $files
else
    coverage report $files
fi
