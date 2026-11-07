import os
import subprocess
import sys


def mfa_environment(root_dir, mfa_env=None):
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = root_dir
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if mfa_env:
        parts = [
            mfa_env,
            os.path.join(mfa_env, "Library", "mingw-w64", "bin"),
            os.path.join(mfa_env, "Library", "usr", "bin"),
            os.path.join(mfa_env, "Library", "bin"),
            os.path.join(mfa_env, "Scripts"),
            os.path.join(mfa_env, "bin"),
        ]
        env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
        env["CONDA_PREFIX"] = mfa_env
    return env


def mfa_executable(mfa_env=None):
    if not mfa_env:
        return "mfa"
    if sys.platform.startswith("win"):
        return os.path.join(mfa_env, "Scripts", "mfa.exe")
    return os.path.join(mfa_env, "bin", "mfa")


def run_mfa(arguments, root_dir, mfa_env=None, log_path=None):
    os.makedirs(root_dir, exist_ok=True)
    command = [mfa_executable(mfa_env)] + [str(a) for a in arguments]
    env = mfa_environment(root_dir, mfa_env)
    print("running:", " ".join(command), flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(command, env=env)
    if result.returncode != 0:
        raise RuntimeError("mfa failed with exit code %d, see %s" % (result.returncode, log_path))
    return result.returncode
