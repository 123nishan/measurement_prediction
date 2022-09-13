input_size=5
hidden_size=128
um_epochs=5
learning_rate=0.001

dutch_demographic_path = "./dutch/demographic_metric.csv"
dutch_measurement_path = "./dutch/measurement_metric.csv"
dutch_extracted_path='./dutch/DutchExtracted.csv'

italy_demographic_path="./italy/ItalyDemographics_csv.csv"
italy_extracted_path='./italy/ItalyExtracted.csv'
# italy_demographic_path="ItalyDemographics.csv"
italy_measurement_path="./italy/ItalyMeasurements_csv.csv"

#demo
subject_number = 'Subject Number'
age = 'Age (Years)'
gender = 'Gender'
height = 'Reported Height (cm)'
weight = 'Reported Weight (kg)'
shoe_size = 'Shoe Size NL'

#measurements
waist = 'Waist Circumference, Pref (mm)'
neck='Neck Base Circumference (mm)'
hip= 'Hip Circumference, Maximum (mm)'
chest='Chest Circumference (mm)'
bust='Bust/Chest Circumference Under Bust (mm)'
crotch_height='Crotch Height (mm)'
shoe_size_italy="Shoe Size IT"
female_chest='Chest Girth (Chest Circumference at Scye) (mm)'
chest_italy='Chest Girth at Scye (Chest Circumference at Scye) (mm)'
upper_chest='Chest Girth (Chest Circumference at Scye) (mm)'
malleolus_outer='Ankle Ht Rt (Malleolus, Lateral) (mm)'
malleolus_inner='Malleolus Med Rt (mm)'
measured_weight='Weight (kg)'
thigh='Thigh Circumference (mm)'
shoulder_breadth='Shoulder Breadth (mm)'
waist_floor='Waist Height, Preferred (mm)'
arm_length='Arm Length (Shoulder to Wrist) (mm)'
italy_upper_chest="Chest Girth at Scye (Chest Circumference at Scye) (mm)"
outer_inseam='Outer Inseam'
inner_inseam='Inner Inseam'
inverted_triangle='Inverted Triangle'
hourglass='Hourglass'
rectangle='Rectangle'
triangle='Triangle'
top_hourglass='Top hourglass'

demographic_column=[subject_number,gender,age, height, weight, shoe_size]
extracted_column=[subject_number,malleolus_inner,malleolus_outer]
measurement_column=[subject_number,waist,chest,neck,hip,crotch_height,thigh,shoulder_breadth,waist_floor,arm_length,upper_chest,measured_weight]

italy_demographic_column=[subject_number,gender,age, height, weight, shoe_size_italy]
italy_measurements_column=[subject_number,waist,chest,neck,hip,crotch_height,thigh,shoulder_breadth,waist_floor,arm_length,italy_upper_chest,measured_weight]

additional_column=[subject_number]

female_measurement_column=[subject_number,waist,chest,bust,neck,hip,crotch_height,thigh,shoulder_breadth,waist_floor,arm_length,upper_chest,measured_weight]
female_italy_measurements_column=[subject_number,waist,chest,bust,neck,hip,crotch_height,thigh,shoulder_breadth,waist_floor,arm_length,italy_upper_chest,measured_weight]

# ,hourglass,inverted_triangle,rectangle,triangle
# ,inverted_triangle,rectangle,triangle
male_inputs_list=[ age, height, weight, shoe_size,inverted_triangle,rectangle,triangle]
female_output_list=[waist,chest,bust,neck,hip,crotch_height,thigh,shoulder_breadth,waist_floor,arm_length,upper_chest,malleolus_outer,malleolus_inner]
female_input_list = [ age, height, weight, shoe_size,hourglass,inverted_triangle,rectangle,triangle]
male_output_list = [ waist,chest,neck,hip,crotch_height,thigh,shoulder_breadth,waist_floor,arm_length,upper_chest,malleolus_outer,malleolus_inner]