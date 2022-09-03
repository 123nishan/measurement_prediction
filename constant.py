input_size=5
hidden_size=128
um_epochs=5
learning_rate=0.001

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

outer_inseam='Outer Inseam'
inner_inseam='Inner Inseam'
inverted_triangle='Inverted Triangle'
hourglass='Hourglass'
rectangle='Rectangle'
triangle='Triangle'

# ,hourglass,inverted_triangle,rectangle,triangle
demographic_male=[ age, height, measured_weight, shoe_size,inverted_triangle,rectangle,triangle]
female_measurement=[waist,chest,bust,neck,hip,crotch_height,thigh,shoulder_breadth,waist_floor,arm_length,upper_chest,malleolus_outer,malleolus_inner]
demographic = [ age, height, weight, shoe_size,hourglass,inverted_triangle,rectangle,triangle]
measurement = [ waist,chest,neck,hip,crotch_height,thigh,shoulder_breadth,waist_floor,arm_length,upper_chest,malleolus_outer,malleolus_inner]