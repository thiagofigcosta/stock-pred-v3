#!/bin/bash

if [[ "$CLEANUP_BEFORE" = [tT][rR][uU][eE] ]] ; then
  /code/clean_up_generated_files.sh --force
fi



if [ -z ${NAS_PARALLELISM+x} ] || [ -z "$NAS_PARALLELISM" ] ; then # keep the +x
  echo "NAS_PARALLELISM is a mandatory env var"
  exit 1
fi
nas_p=`echo $NAS_PARALLELISM | tr -d '"' | tr '[:upper:]' '[:lower:]'`

if [ -z ${LSTM_PARALLELISM+x} ] || [ -z "$LSTM_PARALLELISM" ] ; then # keep the +x
  echo "LSTM_PARALLELISM is a mandatory env var"
  exit 1
fi
lstm_p=`echo $LSTM_PARALLELISM | tr -d '"' | tr '[:upper:]' '[:lower:]'`

if [ -z ${STOCK_LIST+x} ] || [ -z "$STOCK_LIST" ] ; then # keep the +x 
  echo "STOCK_LIST is a mandatory env var"
  exit 1
fi
stocks_array=($(echo "$STOCK_LIST" | tr ',' '\n'))

if [ -z ${START_DATE+x} ] || [ -z "$START_DATE" ] ; then # keep the +x 
  echo "START_DATE is a mandatory env var"
  exit 1
fi
start_date=`echo $START_DATE | tr -d '"' | tr '[:upper:]' '[:lower:]'`

if [ -z ${END_DATE+x} ] || [ -z "$END_DATE" ] ; then # keep the +x 
  echo "END_DATE is a mandatory env var"
  exit 1
fi
end_date=`echo $END_DATE | tr -d '"' | tr '[:upper:]' '[:lower:]'`

if [ -z ${MODE+x} ] || [ -z "$MODE" ] ; then # keep the +x 
  echo "MODE is a mandatory env var"
  exit 1
fi
mode=`echo $MODE | tr -d '"' | tr '[:upper:]' '[:lower:]'`

if [ -z ${NAS_ALG+x} ] || [ -z "$NAS_ALG" ] ; then # keep the +x 
  echo "NAS_ALG is a mandatory env var"
  exit 1
fi
nas_alg=`echo $NAS_ALG | tr -d '"' | tr '[:upper:]' '[:lower:]'`

if [ -z ${NAS_OBJ+x} ] || [ -z "$NAS_OBJ" ] ; then # keep the +x 
  echo "MODE is a mandatory env var"
  exit 1
fi
nas_obj=`echo $NAS_OBJ | tr -d '"' | tr '[:upper:]' '[:lower:]'`

# end mandatory

stock_pred_v3_args="$mode --start=\"$start_date\" --end=\"$end_date\" --nas_p=\"$nas_p\" --lstm_p=\"$lstm_p\" --nas_alg=\"$nas_alg\" --nas_obj=\"$nas_obj\""

if [[ "$DRY_RUN" = [tT][rR][uU][eE] ]] ; then
  stock_pred_v3_args="$stock_pred_v3_args --dry_run"
fi

if [ ! -z ${SS_ID+x} ] && [ ! -z "$SS_ID" ] ; then # keep the +x
  ss_id=`echo $SS_ID | tr -d '"' | tr '[:upper:]' '[:lower:]'`
  stock_pred_v3_args="$stock_pred_v3_args --ss_id=\"$ss_id\""
fi

if [ ! -z ${POP_SZ+x} ] && [ ! -z "$POP_SZ" ] ; then # keep the +x
  pop_sz=`echo $POP_SZ | tr -d '"' | tr '[:upper:]' '[:lower:]'`
  stock_pred_v3_args="$stock_pred_v3_args --pop_sz=\"$pop_sz\""
fi

if [ ! -z ${CHILDREN_SZ+x} ] && [ ! -z "$CHILDREN_SZ" ] ; then # keep the +x
  children_sz=`echo $CHILDREN_SZ | tr -d '"' | tr '[:upper:]' '[:lower:]'`
  stock_pred_v3_args="$stock_pred_v3_args --children_sz=\"$children_sz\""
fi

if [ ! -z ${MAX_EVAL+x} ] && [ ! -z "$MAX_EVAL" ] ; then # keep the +x
  max_eval=`echo $MAX_EVAL | tr -d '"' | tr '[:upper:]' '[:lower:]'`
  stock_pred_v3_args="$stock_pred_v3_args --max_eval=\"$max_eval\""
fi

if [ ! -z ${AGG_METHOD+x} ] && [ ! -z "$AGG_METHOD" ] ; then # keep the +x
  agg_method=`echo $AGG_METHOD | tr -d '"' | tr '[:upper:]' '[:lower:]'`
  stock_pred_v3_args="$stock_pred_v3_args --agg_method=\"$agg_method\""
fi

if [ ! -z ${NAS_REF_DIRS_CONF+x} ] && [ ! -z "$NAS_REF_DIRS_CONF" ] ; then # keep the +x
  nas_ref_dir_conf=`echo $NAS_REF_DIRS_CONF | tr -d '"' | tr '[:upper:]' '[:lower:]'`
  stock_pred_v3_args="$stock_pred_v3_args --nas_ref_dir_conf=\"$nas_ref_dir_conf\""
fi

for stock in "${stocks_array[@]}" ; do 
  el=`echo $el | tr -d '"' | tr '[:upper:]' '[:lower:]'`
  echo "Running stock-pred-v3, the prophet of the stock market for ticker: $stock..."
  eval "python /code/main.py $stock_pred_v3_args --stock=\"$stock\""
  echo "Ran stock-pred-v3, the prophet of the stock market for ticker: $stock!"
done


if [[ "$COMPRESS_AFTER" = [tT][rR][uU][eE] ]] ; then
  filename="/code/experiments/exp_$(date +'%Y-%m-%d-%H%M%S').tar.gz"
  echo "Compressing results and outputs to $filename"
  tar -zcvf "$filename" /code/datasets /code/hyperparameters /code/logs /code/models /code/nas /code/prophets /code/saved_plots
  echo "Compressed results and outputs to $filename!"
fi

tail -f /dev/null # keep running forever