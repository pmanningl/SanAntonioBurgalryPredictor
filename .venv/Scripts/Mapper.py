import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
import customtkinter as ctk
from tkintermapview import TkinterMapView

# Load the trained model
mlp_loaded = joblib.load('mlp_model.pkl')

# Load normalization bounds from CSV
bounds_df = pd.read_csv('bounds_data.csv')
lat_min, lat_max = bounds_df.loc[0, ['lat_min', 'lat_max']]
lon_min, lon_max = bounds_df.loc[0, ['lon_min', 'lon_max']]


def normalize(lat, lon, lat_min, lat_max, lon_min, lon_max):
    lat_normalized = (lat - lat_min) / (lat_max - lat_min)
    lon_normalized = (lon - lon_min) / (lon_max - lon_min)
    return lat_normalized, lon_normalized


def convert_sin_cos_to_time(sin_val, cos_val):
    angle = np.arctan2(sin_val, cos_val)
    angle = angle % (2 * np.pi)
    total_minutes = (angle / (2 * np.pi)) * (24 * 60)
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    return hours, minutes


def predict_time(lat, lon):
    # Normalize latitude and longitude
    lat_normalized, lon_normalized = normalize(lat, lon, lat_min, lat_max, lon_min, lon_max)

    # Predict using the normalized values
    new_data = pd.DataFrame({'Latitude': [lat_normalized], 'Longitude': [lon_normalized]})
    predictions = mlp_loaded.predict(new_data)
    sin_val, cos_val = predictions[0]

    # Convert sin and cos values to time
    hours, minutes = convert_sin_cos_to_time(sin_val, cos_val)
    return f'{hours:02}:{minutes:02}'


class App(ctk.CTk):
    APP_NAME = "TkinterMapView with CustomTkinter"
    WIDTH = 800
    HEIGHT = 500

    SAN_ANTONIO_LAT = 29.4241
    SAN_ANTONIO_LON = -98.4936

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title(App.APP_NAME)
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.minsize(App.WIDTH, App.HEIGHT)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.bind("<Command-q>", self.on_closing)
        self.bind("<Command-w>", self.on_closing)
        self.createcommand('tk::mac::Quit', self.on_closing)

        self.marker_list = []

        # Create two CTkFrames
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = ctk.CTkFrame(master=self, width=150, corner_radius=0, fg_color=None)
        self.frame_left.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        self.frame_right = ctk.CTkFrame(master=self, corner_radius=0)
        self.frame_right.grid(row=0, column=1, rowspan=1, pady=0, padx=0, sticky="nsew")

        # frame_left
        self.frame_left.grid_rowconfigure(2, weight=1)

        self.button_2 = ctk.CTkButton(master=self.frame_left,
                                      text="Clear Markers",
                                      command=self.clear_marker_event)
        self.button_2.grid(pady=(20, 0), padx=(20, 20), row=1, column=0)

        self.map_label = ctk.CTkLabel(self.frame_left, text="Tile Server:", anchor="w")
        self.map_label.grid(row=3, column=0, padx=(20, 20), pady=(20, 0))

        # frame_right
        self.frame_right.grid_rowconfigure(1, weight=1)
        self.frame_right.grid_rowconfigure(0, weight=0)
        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_columnconfigure(1, weight=0)
        self.frame_right.grid_columnconfigure(2, weight=1)

        self.map_widget = TkinterMapView(self.frame_right, corner_radius=0)
        self.map_widget.grid(row=1, rowspan=1, column=0, columnspan=3, sticky="nswe", padx=(0, 0), pady=(0, 0))

        # Set default values
        self.map_widget.set_position(self.SAN_ANTONIO_LAT, self.SAN_ANTONIO_LON)
        self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga", max_zoom=22)
        self.map_widget.set_zoom(10)  # Set the zoom level to 10
        self.map_widget.add_right_click_menu_command(label="=====Add Marker=====", command=self.add_marker_event,
                                                     pass_coords=True)


    def clear_marker_event(self):
        for marker in self.marker_list:
            marker.delete()
        self.marker_list.clear()

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()

    def add_marker_event(self, coords):
        lat, lon = coords
        time_str = predict_time(lat, lon)
        new_marker = self.map_widget.set_marker(lat, lon, text=f"Time: {time_str}")
        self.marker_list.append(new_marker)


if __name__ == "__main__":
    app = App()
    app.start()

