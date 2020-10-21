NODE_ID=0
N_NODES=1
CONFIG=$1
tmpdir='temp'
tmpfile="${NODE_ID}_${CONFIG}"
mkdir -p "${tmpdir}"
cp ${CONFIG}.py ${tmpdir}/${tmpfile}.py
sed -i s/__NODEID__/${NODE_ID}/g ${tmpdir}/${tmpfile}.py
sed -i s/__NNODES__/${N_NODES}/g ${tmpdir}/${tmpfile}.py
python sweep_mc_from_config.py --config ${tmpdir}.${tmpfile}
rm ${tmpdir}/${tmpfile}.py