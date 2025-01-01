from psychopy import visual, event, core
from psychopy.hardware.camera import Camera
import random
import csv
import os
import cv2
import numpy as np

# Setup Experiment and Camera Windows - use dual monitor and adjust the position/size as needed
cap = cv2.VideoCapture(0)

win = visual.Window(size=(1920, 1080), fullscr=True, color=[-1, -1, -1], units='pix', pos=(0, 20), screen=0)
feed_win = visual.Window(size=(1920, 1080) , color=[-1, -1, -1], units='pix', screen=1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    core.quit()

def get_camera_frame():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        core.quit()
    #Convert colour from CV2 BGR to PsychoPy RGB
    frame_adjusted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame to match the PsychoPy window dimensions
    frame_resized = cv2.resize(frame_adjusted, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    return frame_resized

def show_camera_frame():
    frame = get_camera_frame()

    # Convert the frame to a PsychoPy-compatible texture (float32 values, ranging from 0-1)
    texture = visual.ImageStim(feed_win, ori=180, image=frame.astype(np.float32) / 255.0)
    
    texture.draw()
    feed_win.flip()


def show_text_and_wait_for_keypress(text, keys=[]):
    message = visual.TextStim(win, text=text, color=[1, 1, 1], wrapWidth=700)
    event.clearEvents()
    while not keys:
        message.draw()
        win.flip()
        keys = event.getKeys()
        core.wait(0.1) #VERY IMPORTANT: otherwise GPU often crashes
    return keys

def get_typed_input(prompt):
    message = visual.TextStim(win, text=prompt + "\n(Press enter when you are done)", color=[1, 1, 1], wrapWidth=700, pos=(0, 100))
    user_input = []
    while True:
        display_text = prompt + "".join(user_input) + "\n(Press enter when you are done)"
        message.setText(display_text)
        message.draw()
        win.flip()
        keys = event.getKeys()
        if 'return' in keys:  # Enter key to finish input
            break
        elif 'backspace' in keys and user_input:  # Backspace to delete last character
            user_input.pop()
        elif 'escape' in keys:
            core.quit() 
        else:
            user_input += keys  # Add typed characters to input
    return "".join(user_input)

participant_id = get_typed_input("Please enter ID: ")

instructions = [
    "Can You Feel It?\nAre You Being Watched?\n \n Press spacebar to continue.",
    "For each trial, the camera behind you can either be turned on or off.\n  For each trial, you will be asked to provide a response.\n \n Press 'S' to respond 'camera is on' if you feel as if you are being watched.\n Press 'L' to respond 'camera is off' if you feel as if you are not.\n \n Press spacebar to continue.", 
    "For some blocks (sets of multiple trials), you will receive feedback after each trial (whether the camera was in fact on or off), and in other blocks, you will not receive any feedback. \n \nIn each block, your goal is to achieve the highest accuracy (i.e. responding 'on' if the camera was in fact on and vice versa) as possible.\n \n Press spacebar to continue.",
]

for instruction in instructions:
    show_text_and_wait_for_keypress(instruction)

# CSV headers (with underscore convention)
experiment_header = ["participant_id", "block_number", "feedback", "trial_number", "camera_on", "participant_response", "correct_response", "response_correct", "reaction_time", "whole_block_correctness"]

trialLists = []
trialBaseLists = [[0,1,0,1,1,0,0,0,1,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,0,0,0,1,1,0,1,1,1],
    [0,1,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,1,1,0,0,1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,0],
    [1,1,0,0,1,1],
    [1,0,0,0,1,0,1,1,1,0],
    [0,0,0,0,1,0,0,1,1,0,0,1],
    [1,0,1,1,1,1,0,0,0,1,0,1],
    [0,1,0,1,0,1,1,0,1,0,1,0],
    [0,0,0,0,1],
    [0,1,1],
    [1,0]]
trialListLength = 60
trialListLengthNoFeedback = 20

experiment_data = []  # For saving experiment data
block_correct_responses = 0  # Track correct responses per block

def generate_list(baseList, listLength):
    cutList = (baseList*(int(listLength)))[0:listLength]
    return cutList

for baseList in trialBaseLists:
    trialLists.append(generate_list(baseList, trialListLengthNoFeedback))
    trialLists.append(generate_list(baseList, trialListLength))

#Create rectangle to cover camera
black_rect = visual.Rect(feed_win, width=feed_win.size[0], height=feed_win.size[1], fillColor=[-1, -1, -1], lineColor=[-1, -1, -1])

def run_trial(trial_num, block_num, feedback_block, is_On):
    global block_correct_responses
    camera_on = is_On
    response = None
    isRunning = True 
    trial_text = f"Trial #{trial_num}\nProvide response: S (camera is on) or L (camera is off)"
    trial_message = visual.TextStim(win, text=trial_text, color=[1, 1, 1], wrapWidth=700)
    trial_message.draw()
    win.flip()
    # Loop until 's' or 'l' is pressed
    reaction_clock.reset()
    while isRunning:
        if is_On:
            show_camera_frame()
            trial_message.draw()
            win.flip()
        else:
            black_rect.draw()
            feed_win.flip()
            trial_message.draw()
            win.flip()
        keys = event.getKeys() 
        if 's' in keys:
            response = 's'
            isRunning = False 
            black_rect.draw()
            feed_win.flip()
        elif 'l' in keys:
            response = 'l'
            isRunning = False
            black_rect.draw()
            feed_win.flip()
    
    # Get reaction time
    reaction_time = round(reaction_clock.getTime(),4)
    
    # Check if the response is correct
    correct_response = 's' if camera_on else 'l'
    is_correct = (response == correct_response)
    
    if is_correct:
        block_correct_responses += 1
    
    # Log trial data with reaction time
    experiment_data.append([participant_id, block_num, "True" if feedback_block else "False", trial_num, "True" if camera_on else "False", response, correct_response, is_correct, reaction_time, ""])
    
    if feedback_block:
        feedback_text = f"Camera is \n" \
                        f"    Press spacebar to continue"
        feedback_message_status = "on" if camera_on else "off"
        status_color = [0, 1, 0] if camera_on else [1, 0, 0]
        
        feedback_message = visual.TextStim(win, text=feedback_text, color=[1, 1, 1], wrapWidth=700)
        feedback_message_status_text = visual.TextStim(win, text=feedback_message_status, color=status_color, pos=(64, 15))
        feedback_message.draw()
        feedback_message_status_text.draw()
        win.flip()
        event.clearEvents()
        # Wait for the spacebar to be pressed
        keys = []
        while 'space' not in keys:
            keys = event.getKeys()
            core.wait(0.1)  #VERY IMPORTANT: otherwise GPU often crashes
    else:
        # Create the "Response registered" message
        response_message = visual.TextStim(win, text="Response registered", color=[1, 1, 1])
        response_message.draw()
        win.flip()
        core.wait(1)
        keys = []
        keys = event.getKeys()
        

# Run trials for a block
def run_block(block_num, feedback_block, block_sequence):
    global block_correct_responses
    num_trials = len(block_sequence)
    
    block_correct_responses = 0
    # Run the trials in the block
    for i in range(1, num_trials + 1):
        run_trial(i, block_num, feedback_block, block_sequence[i-1])
    
    # Calculate the block correctness as a decimal
    block_correctness = block_correct_responses / num_trials
    
    # Update whole_block_correctness for all trials in this block (for CSV)
    for i in range(-num_trials, 0):
        experiment_data[i][-1] = f"{block_correctness:.2f}"

#Setting up the blocks
totalBlocks = 2*(len(trialBaseLists))
reaction_clock = core.Clock()
for blockNumber in range(1, totalBlocks+1):
    if ((blockNumber+1)%2 == 0):
        show_text_and_wait_for_keypress("Press spacebar to begin block "+str(blockNumber)+".\n"+ "In this block, you will not receive feedback.\n \n Press spacebar to continue.")
        run_block(blockNumber, False, (trialLists[(blockNumber-1)]))
    if ((blockNumber+1)%2 == 1):
        show_text_and_wait_for_keypress("Press spacebar to begin block "+str(blockNumber)+".\n" + "In this block, you will receive feedback. \n \n Press spacebar to continue.")
        run_block(blockNumber, True, (trialLists[(blockNumber-1)]))
        
# Save experiment data to a CSV file
experiment_filename = f'Participant_{participant_id}_ExperimentData.csv'
with open(experiment_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(experiment_header)  # Write header
    writer.writerows(experiment_data)  # Write data rows

show_text_and_wait_for_keypress("Experiment completed. \n Following the trials, participants must answer this brief questionnaire. \n (‘y’ = yes, ‘n’ = no,’d’ = ‘don’t want to answer’). \n Press spacebar to proceed to the questionnaire.")

# Questionnaire
questionnaire = [
    "21, 34, 55, 89.... What is the next number in the sequence? Your answer: \n",
    "Your age: \n",
    "Your gender (male / female / other): ",
    "Are you currently studying or have you studied at a university level? (y/n/d): \n",
    "How would you rate your mathematical abilities? (from 1-10): \n",
    "Did you have a sensation of knowing when you were being watched? (y / n / d): \n",
    "Did you find the experiment difficult? (y / n / d): \n",
    "Did you notice any patterns in the camera being turned on or off? (y / n / d): \n",
    "Would you consider yourself as a religious person? (y / n / d): \n",
    "Do you believe in astrology? (y / n / d): \n",
    "Before this experiment, did you believe in the theory that people can sense when they are being watched? (y / n / d): \n",
    "After this experiment, do you believe in the theory that people can sense when they are being watched? (y / n / d): \n",
    "Are we allowed to use your data from the experiment and the questionnaire (both completely anonymously and GDPR-compliant, of course) to further our research in human cognition? (y / n): \n"
]

# Collect questionnaire responses
questionnaire_responses = []
for question in questionnaire:
    response = get_typed_input(question)
    questionnaire_responses.append(response)

# Save questionnaire responses to a separate CSV file
questionnaire_filename = f'Participant_{participant_id}_QuestionnaireData.csv'
with open(questionnaire_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["question", "response"])
    for question, response in zip(questionnaire, questionnaire_responses):
        writer.writerow([question, response])

# Check if the participant denied the use of their data
if questionnaire_responses[-1] == 'n':
    confirmation = get_typed_input("Are you sure we cannot use your data? Typing 'y' for yes will delete all your data. Type 'n' to go back.")
    if confirmation == 'y':
        # Delete both experiment and questionnaire data files
        if os.path.exists(experiment_filename):
            os.remove(experiment_filename)
        if os.path.exists(questionnaire_filename):
            os.remove(questionnaire_filename)

        # Create a new file indicating that the data was deleted
        with open(f'Participant_{participant_id}_DataDeleted.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["participant", "status"])
            writer.writerow([participant_id, "participant would not save data - all data deleted."])
        
        # Shut down the experiment
        
        _wait_for_keypress("Your data has been deleted. Press spacebar to exit.")
        win.close()
        core.quit()
    
    elif confirmation == 'n':
        # Go back and ask the original consent question again
        new_consent = get_typed_input("Are we allowed to use your data from the experiment and the questionnaire (both anonymously, of course) to further our knowledge on the human brain? (y/n/d): ")
        
        # Update the last response based on their new decision
        questionnaire_responses[-1] = new_consent
        if new_consent == 'y':
            # They consented, proceed without deleting data
            show_text_and_wait_for_keypress("Thank you for allowing us to use your data. Press spacebar to continue.")
        elif new_consent == 'n':
            # Restart the deletion confirmation process
            confirmation = get_typed_input("Are you sure we cannot use your data? Typing 'y' for yes will delete all your data. Type 'n' to go back.")
            if confirmation == 'y':
                # Delete both experiment and questionnaire data files
                if os.path.exists(experiment_filename):
                    os.remove(experiment_filename)
                if os.path.exists(questionnaire_filename):
                    os.remove(questionnaire_filename)

                # Create a new file indicating that the data was deleted
                with open(f'Participant_{participant_id}_DataDeleted.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["participant", "status"])
                    writer.writerow([participant_id, "participant would not save data - all data deleted."])
                
                # Shut down the experiment
                show_text_and_wait_for_keypress("Your data has been deleted. Press spacebar to exit.")
                win.close()
                core.quit()

show_text_and_wait_for_keypress("Thank you for completing the experiment and questionnaire!\n \n Press spacebar to exit.")

# Cleanup: release the camera and close the window
cap.release()
feed_win.close()
win.close()
core.quit()