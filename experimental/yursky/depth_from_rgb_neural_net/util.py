import numpy as np
import tensorflow as tf
from glob import glob
from os import path, remove, rmdir
from shutil import rmtree
from tempfile import mkdtemp
from zipfile import ZipFile


def save(output_file, graph=None, session=None):
    if graph is None:
        graph = tf.get_default_graph()
    if session is None:
        session = tf.get_default_session()
    if session is None:
        session = tf.Session()

    if '.meta' in output_file:
        print('[W] Putting ".meta" in our filename is asking for trouble!')
        return None

    tmp_dir = mkdtemp()
    tmp_output = path.join(tmp_dir, path.basename(output_file))
    with graph.as_default():
        saver = tf.train.Saver(allow_empty=True)
        saver.save(session, tmp_output, write_state=False)

    of = ZipFile(output_file, 'w')
    for f in glob(tmp_output + '.*'):
        of.write(f, path.basename(f))
        remove(f)
    of.close()
    rmdir(tmp_dir)


def load(input_file, graph=None, session=None):
    tmp_dir = mkdtemp()

    f = ZipFile(input_file, 'r')
    f.extractall(tmp_dir)
    f.close()

    # Find the model name
    meta_files = glob(path.join(tmp_dir, '*.meta'))
    if len(meta_files) < 1:
        raise IOError("[E] No meta file found, giving up")
    if len(meta_files) > 1:
        raise IOError("[E] More than one meta file found, giving up")

    meta_file = meta_files[0]
    model_file = meta_file.replace('.meta', '')

    if graph is None:
        graph = tf.get_default_graph()
    if session is None:
        session = tf.get_default_session()
    if session is None:
        session = tf.Session()

    # Load the model in TF
    with graph.as_default():
        saver = tf.train.import_meta_graph(meta_file)
        if saver is not None:
            saver.restore(session, model_file)
    rmtree(tmp_dir)
    return graph


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>" % size
    return strip_def


def show_graph(graph_def, max_const_size=32):
    from IPython.display import display, HTML
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
<script>
function load() {{
document.getElementById("{id}").pbtxt = {data};
}}
</script>
<link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
<div style="height:600px">
<tf-graph-basic id="{id}"></tf-graph-basic>
</div>
""".format(
        data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
<iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
""".format(code.replace('"', '&quot;'))
    display(HTML(iframe))
