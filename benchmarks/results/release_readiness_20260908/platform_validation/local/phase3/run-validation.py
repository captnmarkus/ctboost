from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(__file__).parent
steps={
 "tests":[sys.executable,"-I",str(root/"run-tests.py")],
 "native":["C:/apps/ctboost/.tmp/ctboost-0159-venv/Scripts/ctest.exe","--test-dir",str(root/"build"),"--build-config","Release","--output-on-failure"],
}
name=sys.argv[1]
env=os.environ.copy()
if name=="native":
    env["PYTHONPATH"]=str(root/"release-env/Lib/site-packages")
    env["PATH"]=sys.base_prefix+os.pathsep+env.get("PATH", "")
started=time.time()
with (root/(name+".log")).open("w",encoding="utf-8") as log:
    result=subprocess.run(steps[name],env=env,stdout=log,stderr=subprocess.STDOUT)
(root/(name+"-validation.json")).write_text(json.dumps({"exit_code":result.returncode,"seconds":time.time()-started},indent=2))
print(name,"exit",result.returncode,flush=True)
raise SystemExit(result.returncode)
