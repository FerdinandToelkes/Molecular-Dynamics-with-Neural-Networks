#!/usr/bin/env bash
#
# extract_gradients.sh
#
# Usage: ./extract_gradients.sh gradient 40 >> gradients_au.txt

# if the script is run with more than four arguments, print usage and exit
if [ $# -gt 4 ]; then
    cat <<EOF
This script extracts gradients from an AIMD log file.
It accepts up to two optional arguments:
    1. The first cycle to extract gradients from (default: 1)
    2. The last cycle to extract gradients from (default: 3000)
    3. The input file (default: 'gradient')
    4. The time step in atomic units (au) (default: 40)

The output will be printed to stdout.

Usage: $0 [input_file] [time_step_in_au]
EOF
    exit 1
fi

# set default values for input file and time step
first_cycle="${1:-1}" # optional third argument for the first cycle, default is 1
last_cycle="${2:-3000}" # optional fourth argument for the last cycle, default is 3000
input_file="${3:-gradient}"
time_step="${4:-40}"



# double backslash to escape the backslash in the regex -> otherwise warning from awk
# Allow fortrans-style scientific notation (e.g., -.5D-10) in addition to the standard float format.
awk -v float='[+-]?[0-9]*(\\.[0-9]+)?([dDeE][+-]?[0-9]+)?' -v dt="$time_step" '
    # 1) When we see the cycle line (e.g. "cycle =     24   ex. state energy = ..."), capture and print the corresponding time.
    /^[[:space:]]*cycle[[:space:]]*=[[:space:]]*[0-9]+.*ex\. state energy/ {
        # use match group () to extract the cycle number
        match($0, /cycle[[:space:]]*=[[:space:]]*([0-9]+)/, m)
        # m[0] is the whole match, m[1] is the first group (the cycle number)
        current_cycle = m[1]
        # trim leading/trailing spaces
        gsub(/^ +| +$/, "", current_cycle)
        current_time = (current_cycle - 1) * dt
        printf "t= %s au\n", current_time
        next
    }

    # 2) Look for the first line of three *pure* numbers (no element symbol) after the coordinates block.
    #    We match:
    #      optional space, number (possibly with E-notation), space, number, space, number, optional space, end-of-line
    #    and ensure theres no letter in the line.
    #    The pattern $0 ~ (...) is needed since we are using the variable float defined above.
    $0 ~ ("^[[:space:]]*" float "[[:space:]]+" float "[[:space:]]+" float "[[:space:]]*$") {
        # this is a line in the gradients block, so we print it
        # $1,$2,$3 = gx,gy,gz
        print $1, $2, $3
        # exit 0 # exit after processing the first gradients line
        next
    }

' "$input_file" 


# NOTES
# the gradients file has the following format: 1. block: coordinates (Bohr), 2. block: gradients (Hartree/Bohr), e.g.:

# $grad          cartesian gradients
#   cycle =      1   ex. state energy =    -1113.8476603874 |dE/dxyz| =  0.325922
#    -4.15411425513000      3.89882745056000     -1.25585268049000      o
#     2.57781664749000      4.68726792579000      2.65552588066000      c
#    ...
#   -10.08153476070000     -2.62102374253000      3.00717341632000      h
#   0.54175609906420D-02  -.20144425061444D-01  -.69139185305775D-02
#   -.47222897433367D-02  -.27030735766496D-01  0.27151345903020D-02
#   ...
#   -.32737947341234D-02  0.34894149607213D-01  -.32396844616582D-02
#   cycle =      2   ex. state energy =    -1113.8425570953 |dE/dxyz| =  0.344634
#    -4.14355466824696      3.90564814575682     -1.24800559929469      o
#     2.57701479249884      4.68932894909420      2.65226193543438      c

# this will emit:
#   t= 0.00000000000 au
#   0.54175609906420D-02  -.20144425061444D-01  -.69139185305775D-02
#   -.47222897433367D-02  -.27030735766496D-01  0.27151345903020D-02
#   ...
#   -.32737947341234D-02  0.34894149607213D-01  -.32396844616582D-02
#   ...

# Note for the regexps catching beginning of gradient blocks:
# /^[[:space:]]*                # optional leading whitespace
#   [+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?   # first float: (\.[0-9]+)? -> optional decimal 
#   [[:space:]]+                # at least one separator space
#   [+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?   # second float
#   [[:space:]]*                # optional trailing whitespace
# $/x {
#   … your block start logic …
# }