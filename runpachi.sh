#!/bin/bash

trap 'kill $PID_D 2>/dev/null' EXIT SIGTERM

run_slave () {
	[ "$#" -eq 0 ] && {
		echo "$0: slave cmd: (USING DEFAULT)" >&2
		echo  "./pachi -g localhost:${SLAVE_PORT} -l localhost:${LOG_PORT} -t =5000 -e uct threads=1,slave" >&2
		( nice ./pachi -g localhost:${SLAVE_PORT} -l localhost:${LOG_PORT} -t =5000 -e uct threads=1,slave ) &
	} || {
		echo "$0: slave cmd:" >&2
		echo  "./pachi -g localhost:${SLAVE_PORT} -l localhost:${LOG_PORT} $@" >&2
		( nice ./pachi -g localhost:${SLAVE_PORT} -l localhost:${LOG_PORT} "$@" ) &
	}
	NEWPID=$!
}

run_dist () {
	echo "$0: distributed engine cmd:" >&2
	echo "./pachi -e distributed ${DIST_PORTS}${DIST_SETUP}" >&2
	./pachi -e distributed ${DIST_PORTS}${DIST_SETUP} <&0 &
	NEWPID=$!
}

[ -n "$LOGFILE" ] && exec 2> "$LOGFILE"
[ -z "$NUM_SLAVES" ] && NUM_SLAVES="1"
[ -z "$PACHI_DIR" ] && PACHI_DIR=~/prj/pachi
[ -z "$SLAVE_PORT" ] && SLAVE_PORT=$((10000 + RANDOM % 20000))
[ -z "$LOG_PORT" ] && LOG_PORT=$(($SLAVE_PORT + 1 ))
[ -z "$DIST_SETUP" ] && DIST_SETUP="" #pass_all_alive"

DIST_PORTS="slave_port=${SLAVE_PORT},proxy_port=$LOG_PORT"

cd $PACHI_DIR

cat >&2 <<EOF 

$0: $(date)
LOGFILE=$LOGFILE
NUM_SLAVES=$NUM_SLAVES
PACHI_DIR=$PACHI_DIR
DIST_SETUP=$DIST_SETUP

EOF

# need to separate DIST_PORTS and DIST_SETUP when run_dist
[ -n "$DIST_SETUP" ] && [ ! $( echo "$DIST_SETUP" | grep -e "^," ) ] && DIST_SETUP=",$DIST_SETUP"

run_dist 		#sets NEWPID
echo "$0: started master $NEWPID" >&2
PID_D=$NEWPID
sleep 1
#kill -0 $PID_D 2>/dev/null || { echo "could not start master, exiting" ; exit 1 ; }

SLAVES=
for num in `seq $NUM_SLAVES` ; do
	run_slave "$@"	#sets NEWPID
	SLAVES="$SLAVES $NEWPID"
	echo "$0: started slave $NEWPID" >&2
done

while true ; do
	#echo "" >&2
	## check distributed
	kill -0 $PID_D 2>/dev/null && {
		#echo "$0: master $PID_D alive" >&2
		# running
		NEW_SLAVES=
		for PID in $SLAVES ; do
			kill -0 $PID 2>/dev/null && {
				#echo "$0: slave $PID alive" >&2
				NEWPID=$PID
			} || {
				run_slave		#sets NEWPID
				echo "$0: slave $PID died, restarted as $NEWPID" >&2
			}
			NEW_SLAVES="$NEW_SLAVES $NEWPID"
		done
		SLAVES=$NEW_SLAVES
	} || {
		echo "$0: master $PID_D dead" >&2
		# dead
		echo "$0: killing all the slaves" >&2
		for PID in $SLAVES ; do
			kill -0 $PID 2>/dev/null && {
				echo "$0: killing $PID" >&2
				kill $PID
			} || {
				echo "$0: $PID already dead" >&2
			}
		done
		break
	}
	sleep 2
done

