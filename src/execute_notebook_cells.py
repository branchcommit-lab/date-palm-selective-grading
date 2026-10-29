from pathlib import Path
import argparse,json,contextlib,io,os,sys,traceback,time
import nbformat

def main():
    p=argparse.ArgumentParser(description='Execute notebook Python cells in order in one fresh process, without a network kernel.')
    p.add_argument('notebook',type=Path);p.add_argument('--cwd',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();nb=nbformat.read(a.notebook.resolve(),as_version=4);nbformat.validate(nb);os.chdir(a.cwd)
    ns={'__name__':'__main__'};count=0
    for i,cell in enumerate(nb.cells):
        if cell.cell_type!='code':continue
        cell.outputs=[];stream=io.StringIO();count+=1;cell.execution_count=count
        try:
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):exec(compile(cell.source,f'notebook-cell-{i+1}','exec'),ns)
        except Exception as e:
            cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()),nbformat.v4.new_output('error',ename=type(e).__name__,evalue=str(e),traceback=traceback.format_exc().splitlines())]
            a.output.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,a.output);raise
        cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue())]
    a.output.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,a.output);print(json.dumps({'status':'PASS','code_cells':count,'runner':'ordered Python cell execution in fresh process; not Jupyter kernel execution'}))
if __name__=='__main__':main()
