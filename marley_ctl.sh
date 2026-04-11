#!/bin/bash
# marley_ctl.sh — Marley1 server control
# Usage: marley_ctl.sh {start|stop|restart|status|logs} [--public] [--port N]

SCRIPT="$HOME/marley1/marley_server.py"
PIDFILE="$HOME/marley1/marley.pid"
LOGFILE="$HOME/marley1/logs/marley.log"

mkdir -p "$HOME/marley1/logs"

is_running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null
}

cmd="$1"; shift
extra_args="$@"

case "$cmd" in
  start)
    if is_running; then
      echo "Already running (PID $(cat $PIDFILE))"
      exit 0
    fi
    nohup python3 "$SCRIPT" $extra_args >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 1
    if is_running; then
      echo "Started (PID $(cat $PIDFILE)) — log: $LOGFILE"
    else
      echo "FAILED to start — check $LOGFILE"
      cat "$LOGFILE" | tail -10
      exit 1
    fi
    ;;

  stop)
    if is_running; then
      kill "$(cat $PIDFILE)" && rm -f "$PIDFILE"
      echo "Stopped"
    else
      echo "Not running"
      rm -f "$PIDFILE"
    fi
    ;;

  restart)
    "$0" stop
    sleep 1
    "$0" start $extra_args
    ;;

  status)
    if is_running; then
      PID=$(cat $PIDFILE)
      echo "RUNNING  PID=$PID"
      echo "CMD: $(ps -p $PID -o args= 2>/dev/null)"
      echo "LOG: $LOGFILE"
      # Try to hit the status API
      PORT=$(grep -oP '(?<=--port )\d+' <<< "$extra_args" || echo 7842)
      curl -s "http://100.110.181.128:${PORT}/api/status" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
    else
      echo "STOPPED"
    fi
    ;;

  logs)
    tail -f "$LOGFILE"
    ;;

  *)
    echo "Usage: marley_ctl.sh {start|stop|restart|status|logs} [--public] [--port N]"
    echo ""
    echo "  start            Tailscale-only mode (default)"
    echo "  start --public   Bind to 0.0.0.0 (LAN accessible)"
    echo "  stop             Kill the server"
    echo "  restart          Stop + start"
    echo "  status           Show PID + API status"
    echo "  logs             Tail the log file"
    exit 1
    ;;
esac
