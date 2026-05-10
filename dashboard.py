import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

class EnergyForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Energy Consumption Forecasting Dashboard")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f2f5")

        # Header
        header = tk.Frame(root, bg="#2c3e50", height=80)
        header.pack(fill="x")
        title = tk.Label(header, text="LSTM Energy Consumption Forecasting", font=("Helvetica", 24, "bold"), fg="white", bg="#2c3e50")
        title.pack(pady=20)

        # Main Content
        self.main_frame = tk.Frame(root, bg="#f0f2f5")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Sidebar/Controls
        control_frame = tk.Frame(self.main_frame, bg="#ffffff", width=250, highlightbackground="#d1d9e6", highlightthickness=1)
        control_frame.pack(side="left", fill="y", padx=(0, 20))

        tk.Label(control_frame, text="Metrics Summary", font=("Helvetica", 16, "bold"), bg="white", fg="#2c3e50").pack(pady=10)
        
        # Display Metrics
        metrics_text = (
            "Model: LSTM\n"
            "RMSE: 0.5551 kW\n\n"
            "Decision Tree\n"
            "RMSE: 0.7332 kW\n\n"
            "Random Forest\n"
            "RMSE: 0.4754 kW"
        )
        tk.Label(control_frame, text=metrics_text, font=("Helvetica", 12), bg="white", justify="left").pack(pady=20, padx=10)

        self.btn_load = tk.Button(control_frame, text="Refresh Plots", command=self.load_images, bg="#3498db", fg="white", font=("Helvetica", 12, "bold"), relief="flat", padx=20, pady=10)
        self.btn_load.pack(pady=20)

        # Image Display Area
        self.image_frame = tk.Frame(self.main_frame, bg="#f0f2f5")
        self.image_frame.pack(side="right", fill="both", expand=True)

        self.lbl_plot1 = tk.Label(self.image_frame, bg="#f0f2f5")
        self.lbl_plot1.pack(pady=10)

        self.lbl_plot2 = tk.Label(self.image_frame, bg="#f0f2f5")
        self.lbl_plot2.pack(pady=10)

        self.load_images()

    def load_images(self):
        try:
            # Prediction Plot
            if os.path.exists("prediction_plot.png"):
                img1 = Image.open("prediction_plot.png").resize((600, 300))
                self.photo1 = ImageTk.PhotoImage(img1)
                self.lbl_plot1.config(image=self.photo1)
            
            # Comparison Plot
            if os.path.exists("model_comparison.png"):
                img2 = Image.open("model_comparison.png").resize((600, 300))
                self.photo2 = ImageTk.PhotoImage(img2)
                self.lbl_plot2.config(image=self.photo2)
            else:
                self.lbl_plot2.config(text="Comparison plot not found. Run compare_models.py first.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load images: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyForecastApp(root)
    root.mainloop()
