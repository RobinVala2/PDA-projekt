import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import subprocess

class TrainingUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Main Menu")
        self.window.geometry("200x400")
        
        # Configure grid weights
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Create and setup the main frame
        self.main_frame = ttk.Frame(self.window, padding="20 20 20 20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Initialize variables
        self.map_size = tk.StringVar(value="4x4")
        self.is_slippery = tk.BooleanVar(value=True)
        self.is_training = tk.BooleanVar(value=True)
        self.algo = tk.StringVar(value="GA")
        
        self.widgets = {}
        
        self.create_widgets()
        
    def create_widgets(self):
        # Algorithm Selection
        ttk.Label(self.main_frame, text="Choose algorithm:", font=("Arial", 12)).grid(row=0, column=0, sticky="w", pady=(0, 5))
        algo_pick = ttk.Combobox(self.main_frame, textvariable=self.algo, values=["GA", "RL", "RL penalization"], state="readonly", width=15)
        algo_pick.grid(row=1, column=0, sticky="w", pady=(0, 15))
        algo_pick.bind('<<ComboboxSelected>>', self.on_algo_change)
        
        # Map Size Section
        ttk.Label(self.main_frame, text="Map Size:", font=("Arial", 12)).grid(row=2, column=0, sticky="w", pady=(0, 5))
        map_combo = ttk.Combobox(self.main_frame, textvariable=self.map_size, values=["4x4", "8x8"], state="readonly", width=15)
        map_combo.grid(row=3, column=0, sticky="w", pady=(0, 15))
        
        # Slippery Section
        ttk.Label(self.main_frame, text="Slippery:", font=("Arial", 12)).grid(row=4, column=0, sticky="w", pady=(0, 5))
        ttk.Checkbutton(self.main_frame, variable=self.is_slippery, style='Switch.TCheckbutton').grid(row=5, column=0, sticky="w", pady=(0, 15))
        
        # Episodes Section (RL only)
        episodes_label = ttk.Label(self.main_frame, text="Number of Episodes:", font=("Arial", 12))
        episodes_label.grid(row=6, column=0, sticky="w", pady=(0, 5))
        self.episodes_entry = ttk.Entry(self.main_frame, width=17)
        self.episodes_entry.grid(row=7, column=0, sticky="w", pady=(0, 15))
        
        # Training Section (RL only)
        training_label = ttk.Label(self.main_frame, text="Training:", font=("Arial", 12))
        training_label.grid(row=8, column=0, sticky="w", pady=(0, 5))
        training_button = ttk.Checkbutton(self.main_frame, variable=self.is_training, style='Switch.TCheckbutton')
        training_button.grid(row=9, column=0, sticky="w", pady=(0, 15))
        
        # Store RL-specific widgets for enabling/disabling
        self.widgets['rl'] = [
            episodes_label,
            self.episodes_entry,
            training_label,
            training_button
        ]
        
        # Run Button
        ttk.Button(self.main_frame, text="Run Training", command=self.on_button_click, style='Accent.TButton').grid(row=10, column=0, sticky="ew", pady=(15, 0))
        
        # Configure styles
        self.configure_styles()
        
        # Initial state
        self.on_algo_change(None)

        
    def configure_styles(self):
        # Create custom styles
        style = ttk.Style()
        
        # Configure switch style for checkbuttons
        style.configure('Switch.TCheckbutton', font=("Arial", 10))
        
        # Configure accent style for the main button
        style.configure('Accent.TButton', font=("Arial", 12))

    def on_algo_change(self, event):
        # Enable/disable widgets based on selected algorithm
        for widget in self.widgets['rl']:
            if self.algo.get() == "GA":
                widget.state(['disabled'])
            else:
                widget.state(['!disabled'])
        
    def on_button_click(self):
        try:
            if self.algo.get() == "RL":

                num_episodes = int(self.episodes_entry.get())
                if num_episodes > 20000:
                    messagebox.showwarning("Warning", "Too many episodes to run, please input less than 20000")
                    return
                command = f"python3 q_table_run.py --episodes {num_episodes} --training {self.is_training.get()} --map_name {self.map_size.get()} --is_slippery {self.is_slippery.get()}"
                print(command)
            elif self.algo.get() == "RL penalization":
                num_episodes = int(self.episodes_entry.get())
                if num_episodes > 20000:
                    messagebox.showwarning("Warning", "Too many episodes to run, please input less than 20000")
                    return
                command = f"python3 q_table_run-penalization.py --episodes {num_episodes} --training {self.is_training.get()} --map_name {self.map_size.get()} --is_slippery {self.is_slippery.get()}"
                print(command)
            elif self.algo.get() == "GA":
                command = f"python3 ga_run-slippery{self.is_slippery.get()}.py --map_size {self.map_size.get()}"
                print(command)

            subprocess.run(command, shell=True, check=True, executable='/bin/bash')
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for episodes")
            return
        
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = TrainingUI()
    app.run()