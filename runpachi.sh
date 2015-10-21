#!/bin/bash

trap 'kill $PID_D 2>/dev/null' EXIT SIGINT SIGKILL

run_slave () {
	( nice ./pachi -e uct -g localhost:${SLAVE_PORT} -l localhost:${LOG_PORT} "${SLAVE_SETUP}" ) &
	NEWPID=$!
}

run_dist () {
	./pachi -e distributed ${DIST_PORTS}${DIST_SETUP} 2>> $LOGFILE <&0 &
	NEWPID=$!
}

[ -z "$LOGFILE" ] && LOGFILE="$(pwd)/LOG_pachi"
[ -z "$NUM_SLAVES" ] && NUM_SLAVES="2"
[ -z "$PACHI_DIR" ] && PACHI_DIR=~/prj/pachi
[ -z "$SLAVE_PORT" ] && SLAVE_PORT=1234
[ -z "$LOG_PORT" ] && LOG_PORT=1235
[ -z "$SLAVE_SETUP" ] && SLAVE_SETUP="threads=2,max_tree_size=3072,slave"
[ -z "$DIST_SETUP" ] && DIST_SETUP=""  # pass_all_alive"

# need to separate DIST_PORTS and DIST_SETUP when run_dist
[ -n "$DIST_SETUP" ] && [ ! $( echo "$DIST_SETUP" | grep -e "^," ) ] && DIST_SETUP=",$DIST_SETUP"

DIST_PORTS="slave_port=${SLAVE_PORT},proxy_port=$LOG_PORT"

cd $PACHI_DIR
exec 2> $LOGFILE

cat >&2 <<EOF 

$0: $(date)
LOGFILE=$LOGFILE
NUM_SLAVES=$NUM_SLAVES
PACHI_DIR=$PACHI_DIR
SLAVE_SETUP=$SLAVE_SETUP
DIST_SETUP=$DIST_SETUP

EOF

run_dist 		#sets NEWPID
echo "started master $NEWPID" >&2
PID_D=$NEWPID
sleep 1
#kill -0 $PID_D 2>/dev/null || { echo "could not start master, exiting" ; exit 1 ; }

SLAVES=
for num in `seq $NUM_SLAVES` ; do
	run_slave		#sets NEWPID
	SLAVES="$SLAVES $NEWPID"
	echo "started slave $NEWPID" >&2
done

while true ; do
	#echo "" >&2
	## check distributed
	kill -0 $PID_D 2>/dev/null && {
		#echo "master $PID_D alive" >&2
		# running
		NEW_SLAVES=
		for PID in $SLAVES ; do
			kill -0 $PID 2>/dev/null && {
				#echo "slave $PID alive" >&2
				NEWPID=$PID
			} || {
				run_slave		#sets NEWPID
				echo "slave $PID died, restarted as $NEWPID" >&2
			}
			NEW_SLAVES="$NEW_SLAVES $NEWPID"
		done
		SLAVES=$NEW_SLAVES
	} || {
		echo "master $PID_D dead" >&2
		# dead
		echo "killing all the slaves" >&2
		for PID in $SLAVES ; do
			kill -0 $PID 2>/dev/null && {
				echo "killing $PID" >&2
				kill $PID
			} || {
				echo "$PID already dead" >&2
			}
		done
		break
	}
	sleep 2
done

