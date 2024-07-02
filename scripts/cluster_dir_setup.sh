# this is a quick script to setup directory on the other workers node
source copy.sh gradient_coding_mds/main.py ubuntu w1 ~
source copy.sh gradient_coding_mds/main.py ubuntu w2 ~
source copy.sh gradient_coding_mds/src/ ubuntu w1 ~
source copy.sh gradient_coding_mds/src/ ubuntu w2 ~
