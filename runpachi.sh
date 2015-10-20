#!/bin/bash

trap 'kill $PID_D' EXIT SIGINT SIGKILL

run_slave () {
	( ./pachi -e uct -g localhost:1234 -l localhost:1235 "${SLAVE}${ALL}" ) &
	NEWPID=$!
}

run_dist () {
	./pachi -e distributed ${DIST}${ALL} 2>> ${LOGDIR}/LOG_pachi <&1 &
	NEWPID=$!
}

LOGDIR="$(pwd)"
NUM_SLAVES=2

cd ~/prj/pachi
export ALL=""
export SLAVE="threads=2,max_tree_size=3072,slave"
export DIST="slave_port=1234,proxy_port=1235,pass_all_alive"

run_dist 		#sets NEWPID
echo "started master $NEWPID" >&2
PID_D=$NEWPID

SLAVES=
for num in `seq $NUM_SLAVES` ; do
	run_slave		#sets NEWPID
	SLAVES="$SLAVES $NEWPID"
	echo "started slave $NEWPID" >&2
done

while true ; do
	echo "" >&2
	## check distributed
	kill -0 $PID_D 2>/dev/null && {
		echo "master $PID_D alive" >&2
		# running
		NEW_SLAVES=
		for PID in $SLAVES ; do
			kill -0 $PID 2>/dev/null && {
				echo "slave $PID alive" >&2
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
		sleep 1
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

