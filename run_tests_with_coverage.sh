branch=false
missing=false

while [ "$1" != "" ]; do
    case $1 in
        -b | --branch )          branch=true
                                 ;;
        -m | --show-missing )    missing=true
    esac
    shift
done

if [ "$branch" = true ] ; then
    coverage run --branch unit_tests.py
else
    coverage run unit_tests.py
fi

if [ "$missing" = true ] ; then
    coverage report -m unit_tests.py feature_vector_creation.py
else
    coverage report unit_tests.py feature_vector_creation.py
fi
