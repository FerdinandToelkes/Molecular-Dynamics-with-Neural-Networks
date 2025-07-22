#!/usr/bin/env bash
#
# extract_gradients.sh
#
# Usage: ./extract_gradients.sh input_file [first_cycle] [last_cycle] [time_step] >> gradients.txt

# if the script is run with more than four arguments, print usage and exit
if [ $# -gt 4 ] || [ $# -lt 1 ]; then
    cat <<EOF
This script extracts gradients from an AIMD log file.
It accepts up to four arguments, three of which are optional:
    1. The input file to extract gradients from (required).
    2. The first cycle to extract gradients from (default: 1)
    3. The last cycle to extract gradients from (default: 3000)
    4. The time step in atomic units (au) (default: 40)

The output will be printed to stdout.

Usage: $0 input_file [first_cycle] [last_cycle] [time_step]
EOF
    exit 1
fi

# set default values for all arguments except the first one
input_file="${1}"
first_cycle="${2:-1}" 
last_cycle="${3:-3000}"
time_step="${4:-40}"



# double backslash to escape the backslash in the regex -> otherwise warning from awk
# Allow fortrans-style scientific notation (e.g., -.5D-10) in addition to the standard float format.
awk \
    -v float='[+-]?[0-9]*(\\.[0-9]+)?([dDeE][+-]?[0-9]+)?' \
    -v first_cycle="$first_cycle" \
    -v last_cycle="$last_cycle" \
    -v dt="$time_step" '

    # Initialize a variable to control whether we are extracting gradients
    # and to store the current time.
    BEGIN { extract = 0 }


    # 1) When we see the cycle line (e.g. "cycle =     24   ..."), capture and print the corresponding time.
    /^[[:space:]]*cycle[[:space:]]*=[[:space:]]*[0-9]+.*/ {
        # use match group () to extract the cycle number
        match($0, /cycle[[:space:]]*=[[:space:]]*([0-9]+)/, m)
        # m[0] is the whole match, m[1] is the first group (the cycle number)
        current_cycle = m[1]
        # trim leading/trailing spaces
        gsub(/^ +| +$/, "", current_cycle)

        if (current_cycle >= first_cycle && current_cycle <= last_cycle) {
            extract=1
            current_time = (current_cycle - 1) * dt
            printf "t= %s au\n", current_time 
        }
        else {
            extract=0
        }
        next
        
    }

    # 2) If we are not extracting, skip the rest of the lines
    extract==0 {next}

    # 3) Look for the first line of three *pure* numbers (no element symbol) after the coordinates block.
    #    We match:
    #      optional space, number (possibly with E-notation), space, number, space, number, optional space, end-of-line
    #    and ensure theres no letter in the line.
    #    The pattern $0 ~ (...) is needed since we are using the variable float defined above.
    $0 ~ ("^[[:space:]]*" float "[[:space:]]+" float "[[:space:]]+" float "[[:space:]]*$") {
        # this is a line in the gradients block, so we print it
        # $1,$2,$3 = gx,gy,gz
        print $1, $2, $3
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