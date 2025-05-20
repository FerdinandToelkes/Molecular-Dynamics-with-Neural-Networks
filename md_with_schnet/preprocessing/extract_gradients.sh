#!/usr/bin/env bash
#
# extract_gradients.sh
#
# Usage: ./extract_gradients.sh AIMD_logfile > gradients.out



if [ $# -ne 1 ]; then
    echo "Usage: $0 <AIMD_logfile>"
    exit 1
fi

awk '
    # Extract current time from the first line of the block " t=10240.0000000000"
    /^[[:space:]]*t=[[:space:]]*[0-9]+(\.[0-9]+)?$/ {
        split($0, parts, "=")
        current_time = parts[2]
        # trim leading/trailing spaces
        gsub(/^ +| +$/, "", current_time)  
        printf "t= %s au\n", current_time
        next
    }

    # Whenever we see a line with two floats (time step and "-0.377603333108E-04"),
    # turn on our “in-gradient” flag.
    /^[[:space:]]*[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?[[:space:]]+[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?[[:space:]]*$/ {
        in_grad = 1
        next
    }

    # While in_grad==1, any line that starts with lowercase letters
    # (element symbol) is a gradient line: print it with its element symbol in uppercase.
    in_grad && /^[[:space:]]*[a-z]+[[:space:]]+/ {
        # $1 = element, $2,$3,$4 = gx,gy,gz
        printf "%s  %s  %s  %s\n", toupper($1), $2, $3, $4
        next
    }

    # As soon as we hit something that isn’t an element line, turn off in_grad.
    in_grad {
        in_grad = 0
    }
' "$1"


# NOTES
# For a file like:
# # AIMD log file
# $log
# t=   0.00000000000
#  -3.80479998786       5.08184038405      -1.07244591005     o
#   2.72014086155       3.82774253870       4.31920728517     c
# ...
# -10.6166390575      -4.10691510242     -0.506094990497     h
#   40.0000000000       0.00000000000
# o        -0.252432494869E-03  0.220110101388E-04 -0.213691524510E-03
# c        -0.126156444020E-03 -0.876950460924E-04  0.277167903004E-03
# ...
# h         0.100389920189E-02 -0.759150594462E-03 -0.499024627147E-03
#  0.664008763733E-01  -71.9100189870       0.00000000000
# -0.162356229390E-01  0.110557103632     -0.876102358868E-01
#  0.193098774646E-02 -0.212608677657E-02  0.136526609200E-02
# ...
# -0.524063056678E-01 -0.678309578887E-01  0.277648363794E-02
# t=   40.0000000000
#  -3.81489728765       5.08272082446      -1.08099357103     o
#   2.71509460379       3.82423473686       4.33029400129     c
# ...
# this will emit:
#   t= 0.00000000000 au
#   o  -0.252432494869E-03  0.220110101388E-04  -0.213691524510E-03
#   c  -0.126156444020E-03  -0.876950460924E-04  0.277167903004E-03
#   h   0.100389920189E-02  -0.759150594462E-03  -0.499024627147E-03

# Note for the regexps catching beginning of gradient blocks:
# /^[[:space:]]*                # optional leading whitespace
#   [+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?   # first float: (\.[0-9]+)? -> optional decimal 
#   [[:space:]]+                # at least one separator space
#   [+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?   # second float
#   [[:space:]]*                # optional trailing whitespace
# $/x {
#   … your block start logic …
# }