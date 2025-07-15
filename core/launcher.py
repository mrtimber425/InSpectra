#!/usr/bin/env python
"""
InSpectra Analytics Platform - Unified Cross-Platform Launcher
===============================================================

A comprehensive, single-file launcher that automatically adapts to the user's
environment. It attempts to start a user-friendly GUI, but gracefully falls
back to a console-based interface if a graphical environment is not available.

Features:
- Automatic GUI or console mode detection.
- System environment analysis (Platform, Python, Pip, Memory).
- Dependency checking against 'requirements.txt' and guided installation.
- Detailed progress tracking and activity logging in the GUI.
- One-click application launch.
- Cross-platform compatibility (Windows, macOS, Linux).
"""

import sys
import os
import subprocess
import threading
import queue
import platform
from pathlib import Path
import importlib
import time
from typing import Dict, List, Optional

# --- Dependency Management ---

try:
    from importlib import metadata
except ImportError:
    try:
        import importlib_metadata as metadata
    except ImportError:
        print("Error: Python < 3.8 requires 'importlib_metadata'. Please install it.")
        metadata = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


class LauncherGUI:
    """GUI Launcher for InSpectra Analytics Platform"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("InSpectra Analytics Platform - Launcher")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        self.center_window()

        self.message_queue = queue.Queue()
        self.requirements_status = {}
        self.system_info = {}
        self.installation_in_progress = False

        self.load_requirements()
        self.create_widgets()
        self.start_system_check()
        self.process_queue()

    def center_window(self):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"800x700+{x}+{y}")

    def load_requirements(self):
        self.requirements = []
        try:
            # FIX: Look for requirements.txt in the same directory as the script.
            req_file = Path(__file__).parent / "requirements.txt"
            if req_file.exists():
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            self.requirements.append(line.split('#')[0].strip())
            else:
                self.message_queue.put(
                    ('log', f"requirements.txt not found at {req_file}. Using fallback list.", "warning"))
                self.requirements = [
                    'pandas>=1.5.0', 'numpy>=1.21.0', 'matplotlib>=3.5.0',
                    'seaborn>=0.11.0', 'scikit-learn>=1.1.0', 'duckdb>=0.8.0'
                ]
        except Exception as e:
            print(f"Could not load requirements.txt: {e}")
            self.requirements = []

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        self.create_title_section(main_frame)
        self.create_system_info_section(main_frame)
        self.create_requirements_section(main_frame)
        self.create_log_section(main_frame)
        self.create_control_buttons(main_frame)
        self.create_status_bar(main_frame)

    def create_title_section(self, parent):
        title_frame = ttk.Frame(parent)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        title_frame.grid_columnconfigure(0, weight=1)
        tk.Label(title_frame, text="🚀 InSpectra Analytics Platform", font=('Arial', 18, 'bold'), fg='#2c3e50').grid(
            row=0, column=0)
        tk.Label(title_frame, text="Launcher & Environment Manager", font=('Arial', 12), fg='#34495e').grid(row=1,
                                                                                                            column=0,
                                                                                                            pady=(5, 0))
        tk.Label(title_frame, text="This launcher checks your environment, installs dependencies, and starts the app.",
                 font=('Arial', 10), fg='#7f8c8d', justify=tk.CENTER).grid(row=2, column=0, pady=(10, 0))

    def create_system_info_section(self, parent):
        sys_frame = ttk.LabelFrame(parent, text="System Information", padding="10")
        sys_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        sys_frame.grid_columnconfigure(1, weight=1)
        info_items = [("🖥️ Platform:", "platform"), ("🐍 Python:", "python_version"), ("📦 Pip:", "pip_version"),
                      ("💾 Memory:", "memory"), ("📁 Directory:", "working_dir")]
        self.sys_labels = {}
        for i, (label, key) in enumerate(info_items):
            tk.Label(sys_frame, text=label, font=('Arial', 9, 'bold')).grid(row=i, column=0, sticky=tk.W, padx=(0, 10),
                                                                            pady=2)
            self.sys_labels[key] = tk.Label(sys_frame, text="Checking...", font=('Arial', 9), fg='#7f8c8d')
            self.sys_labels[key].grid(row=i, column=1, sticky=tk.W, pady=2)

    def create_requirements_section(self, parent):
        req_frame = ttk.LabelFrame(parent, text="Requirements Status", padding="10")
        req_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        req_frame.grid_columnconfigure(0, weight=1)
        tree_frame = ttk.Frame(req_frame)
        tree_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # *** FIX: Define the 'Package' column and set show='headings' ***
        self.req_tree = ttk.Treeview(tree_frame, columns=('Package', 'Status', 'Version', 'Required'), show='headings',
                                     height=8)
        self.req_tree.heading('Package', text='Package')
        self.req_tree.heading('Status', text='Status')
        self.req_tree.heading('Version', text='Installed')
        self.req_tree.heading('Required', text='Required')
        self.req_tree.column('Package', width=180, anchor='w')
        self.req_tree.column('Status', width=100, anchor='w')
        self.req_tree.column('Version', width=100, anchor='w')
        self.req_tree.column('Required', width=100, anchor='w')

        self.req_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.req_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.req_tree.configure(yscrollcommand=tree_scroll.set)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(req_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

    def create_log_section(self, parent):
        log_frame = ttk.LabelFrame(parent, text="Activity Log", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=12, font=('Consolas', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.tag_configure('info', foreground='#2c3e50')
        self.log_text.tag_configure('success', foreground='#27ae60', font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure('warning', foreground='#f39c12', font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure('error', foreground='#e74c3c', font=('Consolas', 9, 'bold'))

    def create_control_buttons(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        left_frame = ttk.Frame(button_frame);
        left_frame.grid(row=0, column=0, sticky=tk.W)
        self.refresh_btn = ttk.Button(left_frame, text="🔄 Refresh Check", command=self.start_system_check)
        self.refresh_btn.grid(row=0, column=0, padx=(0, 10))
        self.install_btn = ttk.Button(left_frame, text="📦 Install Requirements", command=self.install_requirements,
                                      state='disabled')
        self.install_btn.grid(row=0, column=1, padx=(0, 10))
        right_frame = ttk.Frame(button_frame);
        right_frame.grid(row=0, column=1, sticky=tk.E)
        button_frame.grid_columnconfigure(1, weight=1)
        self.launch_btn = ttk.Button(right_frame, text="🚀 Launch App", command=self.launch_application,
                                     state='disabled')
        self.launch_btn.grid(row=0, column=0, padx=(0, 10))
        self.exit_btn = ttk.Button(right_frame, text="❌ Exit", command=self.root.quit)
        self.exit_btn.grid(row=0, column=1)

    def create_status_bar(self, parent):
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=5, column=0, sticky=(tk.W, tk.E))
        status_frame.grid_columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Starting system check...")
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, font=('Arial', 9), fg='#7f8c8d',
                                     anchor='w')
        self.status_label.grid(row=0, column=0, sticky=tk.W)

    def log_message(self, message: str, level: str = 'info'):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", level)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, status: str):
        self.status_var.set(status)
        self.root.update_idletasks()

    def start_system_check(self):
        if self.installation_in_progress: return
        self.log_message("Starting comprehensive system check...", 'info')
        self.update_status("Checking system environment...")
        for item in self.req_tree.get_children(): self.req_tree.delete(item)
        self.requirements_status.clear()
        thread = threading.Thread(target=self.check_system_thread, daemon=True)
        thread.start()

    def check_system_thread(self):
        try:
            self.message_queue.put(('system_info', gather_system_info()))
            self.message_queue.put(('status', 'Checking package requirements...'))
            total_reqs = len(self.requirements)
            if total_reqs == 0:
                self.message_queue.put(('log', "No requirements to check.", 'warning'))

            for i, requirement in enumerate(self.requirements):
                self.message_queue.put(('progress', ((i + 1) / total_reqs) * 100 if total_reqs > 0 else 100))
                result = check_requirement(requirement)
                self.message_queue.put(('requirement', requirement, result))
                time.sleep(0.05)

            self.message_queue.put(('progress', 100))
            self.message_queue.put(('check_complete', None))
        except Exception as e:
            self.message_queue.put(('error', f"System check failed: {e}"))

    def process_queue(self):
        try:
            while True:
                message_type, *args = self.message_queue.get_nowait()
                if message_type == 'status':
                    self.update_status(args[0])
                elif message_type == 'log':
                    self.log_message(args[0], args[1] if len(args) > 1 else 'info')
                elif message_type == 'system_info':
                    self.update_system_info(args[0])
                elif message_type == 'requirement':
                    self.update_requirement_display(args[0], args[1])
                elif message_type == 'progress':
                    self.progress_var.set(args[0])
                elif message_type == 'check_complete':
                    self.on_check_complete()
                elif message_type == 'installation_complete':
                    self.on_installation_complete()
                elif message_type == 'error':
                    self.log_message(args[0], 'error'); self.update_status("An error occurred")
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def update_system_info(self, info: Dict[str, str]):
        for key, value in info.items():
            if key in self.sys_labels: self.sys_labels[key].config(text=value)
        self.system_info = info

    def update_requirement_display(self, requirement: str, result: Dict[str, str]):
        package_name = requirement.split('>=')[0].strip()
        icon = "✅" if "✅" in result['status'] else "⚠️" if "⚠️" in result['status'] else "❌" if "❌" in result[
            'status'] else "❓"
        # *** FIX: Insert values in the correct order matching the defined columns ***
        self.req_tree.insert('', 'end', values=(
        f"{icon} {package_name}", result['status'], result['installed_version'], result['required_version']))
        self.requirements_status[package_name] = result

    def on_check_complete(self):
        missing_count = sum(1 for r in self.requirements_status.values() if "❌" in r['status'])
        if missing_count == 0:
            self.log_message("✅ All requirements satisfied! Ready to launch.", 'success')
            self.update_status("Ready to launch Application")
            self.launch_btn.config(state='normal')
            self.install_btn.config(state='disabled')
        else:
            self.log_message(f"⚠️ {missing_count} requirements missing. Click 'Install Requirements' to fix.",
                             'warning')
            self.update_status(f"{missing_count} requirements missing")
            self.launch_btn.config(state='disabled')
            self.install_btn.config(state='normal')

    def install_requirements(self):
        if self.installation_in_progress: return
        missing_reqs = [req for req, res in self.requirements_status.items() if "❌" in res['status']]
        if not missing_reqs:
            messagebox.showinfo("Info", "No missing requirements to install!")
            return
        if not messagebox.askyesno("Install Requirements",
                                   f"This will install {len(missing_reqs)} missing packages. Continue?"):
            return
        self.installation_in_progress = True
        self.install_btn.config(state='disabled', text="Installing...")
        self.log_message("Starting package installation...", 'info')
        thread = threading.Thread(target=self.install_packages_thread, args=(missing_reqs,), daemon=True)
        thread.start()

    def install_packages_thread(self, packages: List[str]):
        try:
            total = len(packages)
            for i, pkg in enumerate(packages):
                self.message_queue.put(('status', f"Installing {pkg}... ({i + 1}/{total})"))
                original_req = next((r for r in self.requirements if r.startswith(pkg)), pkg)
                try:
                    result = subprocess.run([sys.executable, '-m', 'pip', 'install', original_req, '--upgrade'],
                                            capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        self.message_queue.put(('log', f"✅ {pkg} installed successfully", 'success'))
                    else:
                        self.message_queue.put(('log', f"❌ Failed to install {pkg}: {result.stderr}", 'error'))
                except Exception as e:
                    self.message_queue.put(('log', f"❌ Error installing {pkg}: {e}", 'error'))
                self.message_queue.put(('progress', ((i + 1) / total) * 100))
            self.message_queue.put(('installation_complete', None))
        except Exception as e:
            self.message_queue.put(('log', f"Installation process failed: {e}", 'error'))
            self.message_queue.put(('installation_complete', None))

    def on_installation_complete(self):
        self.installation_in_progress = False
        self.install_btn.config(state='normal', text="📦 Install Requirements")
        self.log_message("Installation process completed. Rechecking requirements...", 'info')
        self.root.after(500, self.start_system_check)

    def launch_application(self):
        self.log_message("🚀 Launching Application...", 'success')
        self.update_status("Starting application...")
        try:
            # FIX: Look for main_gui.py in the same directory as the script.
            main_script = Path(__file__).parent / "main_gui.py"
            if not main_script.exists():
                raise FileNotFoundError(f"main_gui.py not found at {main_script}")

            self.root.withdraw()
            proc = subprocess.Popen([sys.executable, str(main_script)], cwd=main_script.parent)
            proc.wait()
            self.root.quit()

        except Exception as e:
            self.log_message(f"❌ Failed to launch application: {e}", 'error')
            messagebox.showerror("Launch Error", f"Could not start application:\n\n{e}")
            self.update_status("Launch failed. Please check the log.")
            self.root.deiconify()

    def run(self):
        self.root.mainloop()


# --- Console Launcher and Helper Functions ---

def gather_system_info() -> Dict[str, str]:
    info = {}
    system = platform.system()
    try:
        if system == "Windows":
            info['platform'] = f"Windows {platform.release()}"
        elif system == "Darwin":
            info['platform'] = f"macOS {platform.mac_ver()[0]}"
        elif system == "Linux":
            try:
                import distro; info['platform'] = f"{distro.name()} {distro.version()}"
            except ImportError:
                info['platform'] = f"Linux {platform.release()}"
        else:
            info['platform'] = f"{system} {platform.release()}"
        info[
            'python_version'] = f"{platform.python_implementation()} {sys.version.split()[0]} ({platform.architecture()[0]})"
        try:
            res = subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True, timeout=5)
            info['pip_version'] = res.stdout.strip().split()[1] if res.returncode == 0 else "Not available"
        except Exception:
            info['pip_version'] = "Not available"
        if psutil:
            mem = psutil.virtual_memory()
            info['memory'] = f"{mem.total / (1024 ** 3):.1f} GB total, {mem.available / (1024 ** 3):.1f} GB available"
        else:
            info['memory'] = "psutil not installed"
        info['working_dir'] = str(Path.cwd())
    except Exception as e:
        print(f"Warning: Could not gather all system info: {e}")
    return info


def check_requirement(requirement: str) -> Dict[str, str]:
    if not metadata: return {'status': "❓ Error", 'installed_version': "Cannot check", 'required_version': "N/A"}
    try:
        req_name = requirement.split('>=')[0].strip()
        req_version_spec = requirement.split('>=')[1].strip() if '>=' in requirement else "any"

        if req_name.lower() == 'scikit-learn': req_name = 'scikit-learn'

        try:
            installed_version = metadata.version(req_name)
            if req_version_spec != "any":
                from packaging import version
                if version.parse(installed_version) >= version.parse(req_version_spec):
                    return {'status': "✅ OK", 'installed_version': installed_version,
                            'required_version': req_version_spec}
                else:
                    return {'status': "⚠️ Old Version", 'installed_version': installed_version,
                            'required_version': req_version_spec}
            return {'status': "✅ OK", 'installed_version': installed_version, 'required_version': req_version_spec}
        except metadata.PackageNotFoundError:
            return {'status': "❌ Missing", 'installed_version': "Not installed", 'required_version': req_version_spec}
    except Exception as e:
        return {'status': "❓ Error", 'installed_version': f"Check failed: {e}", 'required_version': "N/A"}


def run_console_launcher():
    print("🚀 InSpectra Analytics Platform - Console Launcher")
    print("=" * 50)

    info = gather_system_info()
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("-" * 50)

    # FIX: Look for requirements.txt in the same directory as the script.
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print(f"⚠️ Warning: requirements.txt not found at {req_file}. Cannot check dependencies.")
    else:
        print("Checking requirements...")
        with open(req_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        missing_reqs = []
        for req in requirements:
            result = check_requirement(req)
            print(f"  {req:<25} Status: {result['status']}")
            if result['status'] == "❌ Missing":
                missing_reqs.append(req)

        if missing_reqs:
            print("\n❌ Some requirements are missing.")
            install = input("Do you want to try and install them now? (y/n): ").lower()
            if install == 'y':
                for req in missing_reqs:
                    print(f"Installing {req}...")
                    subprocess.run([sys.executable, '-m', 'pip', 'install', req])
        else:
            print("✅ All requirements are met.")

    print("\n🚀 Attempting to launch the application...")
    # FIX: Look for main_gui.py in the same directory as the script.
    main_script = Path(__file__).parent / "main_gui.py"
    if not main_script.exists():
        print(f"❌ Error: main_gui.py not found at {main_script}. Cannot launch.")
        return

    try:
        subprocess.run([sys.executable, str(main_script)], check=True, cwd=main_script.parent)
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")


# --- Main Execution Logic ---

def main():
    """Main entry point for the unified launcher."""
    force_console = "--console" in sys.argv

    if not force_console and TKINTER_AVAILABLE:
        print("🖼️  Tkinter found. Starting GUI launcher...")
        try:
            app = LauncherGUI()
            app.run()
        except Exception as e:
            print(f"❌ GUI launcher failed: {e}")
            print("\n🖥️  Falling back to console mode...")
            run_console_launcher()
    else:
        if force_console:
            print("🖥️  Console mode requested.")
        else:
            print("⚠️  Tkinter not found. Starting in console mode.")
        run_console_launcher()


if __name__ == "__main__":
    # This ensures that imports within your project work correctly
    sys.path.insert(0, str(Path(__file__).parent))

    if sys.version_info < (3, 8):
        print(f"❌ FATAL: Python 3.8+ is required, but you are using {sys.version_info.major}.{sys.version_info.minor}.")
        sys.exit(1)

    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Launcher interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        if TKINTER_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Fatal Launcher Error", str(e))
        sys.exit(1)
