#!/usr/bin/env bash
#
# extract_gradients.sh
#
# Usage: ./extract_gradients.sh AIMD_logfile >> gradients.txt


# if the script is not run with exactly one argument, print usage and exit
if [ $# -ne 1 ]; then
    echo "This script extracts gradients from an AIMD log file."
    echo "It expects the log file to be passed as an argument."
    echo "The output will be printed to stdout."
    echo "Usage: $0 <AIMD_logfile>"
    exit 1
fi

# double backslash to escape the backslash in the regex -> otherwise warning from awk
awk -v float='[+-]?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?' '
    # 1) When we see the time-stamp line (e.g. "t=   0.00000000000"), capture and print it.
    #    Reset our state so we know a new timestep has begun.
    #    (energy_line_seen ensures we skip the "energy" line once per block.)
    /^[[:space:]]*t=[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]*$/ {
        # pull out the number after "="
        split($0, parts, "=")
        current_time = parts[2]
        # trim leading/trailing spaces
        gsub(/^ +| +$/, "", current_time)  
        printf "t= %s au\n", current_time
        energy_line_seen = 0
        next
    }

    # 2) Look for the first line of three *pure* numbers (no element symbol) after the velocities block.
    #    That first 3 number line is the "energy" line we want to SKIP, so we mark force_block_seen.
    #    The *next* 3-number line is the start of the gradients proper.
    #    We match:
    #      optional space, number (possibly with E-notation), space, number, space, number, optional space, end-of-line
    #    and ensure theres no letter in the line.
    $0 ~ ("^[[:space:]]*" float "[[:space:]]+" float "[[:space:]]+" float "[[:space:]]*$") {
        if (!energy_line_seen) {
            # this is the energy line, so we skip it
            energy_line_seen = 1
        }
        else {
            # this is a line in the gradients block, so we print it
            # $1,$2,$3 = gx,gy,gz
            print $1, $2, $3
        }
        next
    }

' "$1" # "$1" is the input file passed as an argument to the script


# NOTES
# the mdlog.i file has the following format: 1. block: coordinates (Bohr), 2. block: velocities (Bohr/t_au), 3. block: gradients (Hartree/Bohr), e.g.:

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
#  0.664008763733E-01  -71.9100189870       0.00000000000            <-- some properties like energy (ignore)
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
# -0.162356229390E-01  0.110557103632     -0.876102358868E-01
#  0.193098774646E-02 -0.212608677657E-02  0.136526609200E-02
#  ...
# -0.524063056678E-01 -0.678309578887E-01  0.277648363794E-02

# Note for the regexps catching beginning of gradient blocks:
# /^[[:space:]]*                # optional leading whitespace
#   [+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?   # first float: (\.[0-9]+)? -> optional decimal 
#   [[:space:]]+                # at least one separator space
#   [+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?   # second float
#   [[:space:]]*                # optional trailing whitespace
# $/x {
#   … your block start logic …
# }