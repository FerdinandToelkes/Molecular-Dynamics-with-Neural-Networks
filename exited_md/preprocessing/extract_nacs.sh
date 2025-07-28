#!/usr/bin/env bash
#
# extract_nacs.sh
#
# Usage: ./extract_nacs.sh input_file [time_step] >> nacs.txt

# if the script is run with more than four arguments, print usage and exit
if [ $# -gt 4 ] || [ $# -lt 1 ]; then
    cat <<EOF
This script extracts non-adiabatic couplings (NACs) from an AIMD log file.
It accepts up to four arguments, three of which are optional:
    1. The input file to extract NACs from (required).
    2. The time step in atomic units (au) (default: 40)

The output will be printed to stdout.

Usage: $0 input_file [time_step]
EOF
    exit 1
fi

# set default value for second argument
input_file="${1}"
time_step="${2:-40}"

# check if the input file exists and is readable
if [[ ! -r "$input_file" ]]; then
  echo "Error: cannot read '$input_file'." >&2
  exit 2
fi

# double backslash to escape the backslash in the regex -> otherwise warning from awk
# Allow fortrans-style scientific notation (e.g., -.5D-10) in addition to the standard float format.
awk \
    -v float='[+-]?[0-9]*(\\.[0-9]+)?([dDeE][+-]?[0-9]+)?' \
    -v dt="$time_step" '

    # Initialize a variable to control whether we are extracting nacs
    BEGIN { in_info_block = 1 }

    # 1) If the line starts with a $, we are (again) in the info block.
    #    We set the in_info_block variable to 1 and skip the line.
    /^[[:space:]]*\$/ {
        in_info_block = 1
        next
    }

    # 2) When we see the cycle line (e.g. "cycle =     24   ..."), capture and print the corresponding time.
    /^[[:space:]]*cycle[[:space:]]*=[[:space:]]*[0-9]+.*/ {
        # use match group () to extract the cycle number
        match($0, /cycle[[:space:]]*=[[:space:]]*([0-9]+)/, m)
        
        # m[0] is the whole match, m[1] is the first group (the cycle number)
        current_cycle = m[1]
        
        # trim leading/trailing spaces
        gsub(/^ +| +$/, "", current_cycle)

        current_time = (current_cycle - 1) * dt
        printf "t= %s au\n", current_time 
        
        # we are no longer in the info block
        in_info_block = 0  
        
        # do nothing else for this line
        next 
        
    }

    # 3) If we are in the info block, skip lines until we find the start of the nacs block.
    in_info_block {next}

    # 4) Look for the first line of three *pure* numbers (no element symbol) after the coordinates block.
    #    We match:
    #      optional space, number (possibly with E-notation), space, number, space, number, optional space, end-of-line
    #    and ensure theres no letter in the line.
    #    The pattern $0 ~ (...) is needed since we are using the variable float defined above.
    $0 ~ ("^[[:space:]]*" float "[[:space:]]+" float "[[:space:]]+" float "[[:space:]]*$") {
        # this is a line in the nacs block, so we print it; $1,$2,$3 = nacx,nacy,nacz
        # replace D with E for Fortran-style scientific notation
        gsub(/[dD]/, "E", $1)  
        gsub(/[dD]/, "E", $2)
        gsub(/[dD]/, "E", $3)

        # print the three numbers
        print $1, $2, $3

        # do nothing else for this line
        next
    }
    
    

' "$input_file" 


# NOTES
# the control file has the following format: Some information line and then 1. block: coordinates (Bohr), 2. block: nacs (1/Bohr), e.g.:


# $title
# $symmetry c1
# ...
# $dft
#    functional pbe0
#    gridsize   m3
#   weight derivatives
# ...
# $nacme
# $nac            cartesian nacs
#   cycle =      1 nacme ex. state energy =    -1113.8476603874 |dE/dxyz| =  0.325922
#    -4.15411425513000      3.89882745056000     -1.25585268049000      o
#     2.57781664749000      4.68726792579000      2.65552588066000      c
#    ...
#   -10.08153476070000     -2.62102374253000      3.00717341632000      h
#    0.29826861422769D-01  -.53748529354036D-01  -.50782975915401D-01
#    0.15231581674648D-01  0.12623246909364D+00  -.26482479683289D-01
#   ...
#   0.10520656746940D-04  0.77935034961295D-03  -.11454725527043D-02
#   cycle =      2 nacme ex. state energy =    -1113.8425570953 |dE/dxyz| =  0.344634
#    -4.14355466824696      3.90564814575682     -1.24800559929469      o
#     2.57701479249884      4.68932894909420      2.65226193543438      c

# this will emit:
#   t= 0.00000000000 au
#    0.29826861422769D-01  -.53748529354036D-01  -.50782975915401D-01
#    0.15231581674648D-01  0.12623246909364D+00  -.26482479683289D-01
#   ...
#   0.10520656746940D-04  0.77935034961295D-03  -.11454725527043D-02
#   ...

# Note for the regexps catching beginning of nacs blocks:
# /^[[:space:]]*                # optional leading whitespace
#   [+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?   # first float: (\.[0-9]+)? -> optional decimal 
#   [[:space:]]+                # at least one separator space
#   [+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?   # second float
#   [[:space:]]*                # optional trailing whitespace
# $/x {
#   … your block start logic …
# }