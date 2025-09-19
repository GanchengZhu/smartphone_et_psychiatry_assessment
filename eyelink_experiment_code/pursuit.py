#!/usr/bin/env python3
# encoding: utf-8
# The task to test the Lissajous curve of the romantic relationship
# import relevant modules

# _________________________import relevant modules__________________________

import pylink
import os
import random
from math import pi, sin
from psychopy import visual, event, core, gui, monitors
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

# __________________________def relevant functions__________________________
# use a dialog to record basic info of participants
def get_sub_info():
    """this is a function aiming at collecting subjects' basic information"""

    task_dlg = gui.Dlg(title=u'平滑追踪任务')
    task_dlg.addField(u'序号：')
    task_dlg.addField(u'Session：')
    task_dlg.addField(u'年龄：')
    task_dlg.addField(u'姓名缩写：')
    ok_data = task_dlg.show()
    if task_dlg.OK:
        return ok_data


def quit_function():
    """function of quiting the experiment """

    win.close()
    core.quit()


def clear_screen(win, bgColor):
    """ clear up the PsychoPy window"""

    win.fillColor = bgColor
    win.flip()


# __________________________set relevant settings___________________________
# show the dialog to collect the info
sub_info_org = get_sub_info()

# if 'control+q' were pressed, quit the experiment
event.globalKeys.add(key='q', modifiers=['ctrl'], func=quit_function)

# Create a custom monitor object to store monitor information
customMon = monitors.Monitor('ASUS144HZ', width=53, distance=75)
customMon.setSizePix([1920, 1080])

# define the experiment mouse and window
win = visual.Window(fullscr=True, monitor=customMon, units='pix' , screen=1)

# hide the mouse curse
win.mouseVisible = False

# set the screen resolution
SCN_W, SCN_H = win.size

# ____________________set the settings to the Eyelink_______________________
# Step 1: Connect to the tracker
tk = pylink.EyeLink('100.1.1.1')

# Step2: open an EDF data file on th EyeLink Host PC
# the file name here should match the name of the file received later
edf_file = 'ps_' + sub_info_org[0] + '_' + sub_info_org[1] + '.edf'
tk.openDataFile(edf_file)
# Optional file header
tk.sendCommand("add_file_preamble_text 'Romantic Freeview task'")

# __________set the Eyelink in offline mode and change some pars____________
# setup Host parameters
# Put the tracker in offline mode before we change its parameters
tk.setOfflineMode()
pylink.msecDelay(50)

# Set the sampling rate to 500Hz
tk.sendCommand("sample_rate 1000")

# pass the screen resolution to the tracker
tk.sendCommand(f"screen_pixel_coords = 0 0 {SCN_W-1} {SCN_H-1}")

# Record a DISPLAY_SCREEN message to let Data Viewer know the
# correct screen resolution to use when visualizing the data
tk.sendMessage(f"DISPLAY_COORDS = 0 0 {SCN_W-1} {SCN_H-1}")

# Set the calibration type to 9-point (HV9)
tk.sendCommand("calibration_type = HV9")

# Set calibration and validation area 
tk.sendCommand("calibration_area_proportion = 0.65 0.90")
tk.sendCommand("validation_area_proportion = 0.65 0.90")

# ___________________define the function of a single trial__________________
# prepare the pursuit target, the clock and the movement parameters
target = visual.GratingStim(win, tex=None, mask='circle', size=25)
pursuitClock = core.Clock()
# set the parameter of the curve
amp_x, amp_y = (SCN_W * 0.246/2, SCN_H * 0.881/2)  # match the phone screen in dva
freq_x, freq_y = (1 / 8.0, 1 / 12.0)
trial_duration = 24

#phi_list = [3/2,1/2]
phi_list = [0]

# ___________________define the function of a single trial__________________
def run_a_trial(phi):
    """ Run a smooth pursuit trial

    arguments:
        trial_duration: the duration of the pursuit movement
    The following equation defines a sinusoidal movement pattern
    y(t) = amplitude * sin(2 * pi * frequency * t + phase)
    for circular or elliptic movements, the phase in x and y directions
    should be pi/2 (direction matters)."""
    phase_x, phase_y = (pi * phi, 0)
    
    bgcolor_RGB = (128, 128, 128)

    # put the tracker in the offline mode first
    tk.setOfflineMode()

    # record_status_message : show some info on the Host PC
    # here we show which task is performaing
    tk.sendCommand("record_status_message '%s'" % 'lissajous')


   # drift check
    # the doDriftCorrect() function requires target position in integers
    # the last two arguments:
    # draw_target (1-default, 0-draw the target then call doDriftCorrect)
    # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
    #
    # drift-check and re-do camera setup if ESCAPE is pressed
    target = visual.GratingStim(win, tex=None, mask='circle', size=23)
    tar_x = amp_x*sin(phase_x)
    tar_y = amp_y*sin(phase_y)
    target.pos = (tar_x, tar_y)
    while True:
        # set the lissajous curve related functions and target start position
        # and draw the target
        target.draw()
        win.flip()
        try:
            error = tk.doDriftCorrect(int(SCN_W/2 + tar_x),int(SCN_H/2 - tar_y), 0, 1)
            # break following a success drift-check
            if error is not pylink.ESC_KEY:
                break
        except:
            pass

    # put tracker in idle/offline mode before recording
    tk.setOfflineMode()
    # start recording
    pylink.pumpDelay(50)
    tk.startRecording(1, 1, 1, 1)

    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)

    # Send a message to mark movement onset 
    frame = 0
    while True:
        target.pos = (tar_x, tar_y)
        target.draw()
        win.flip()
        flip_time = core.getTime()
        frame += 1
        if frame == 1:
            tk.sendMessage('movement_onset')
            move_start = core.getTime() 
        else:
            _x = int(tar_x + SCN_W/2.0)
            _y = int(SCN_H/2.0 - tar_y)
            tar_msg = f'!V TARGET_POS target {_x}, {_y} 1 0'
            tk.sendMessage(tar_msg)
            draw_msg = f'!V FIXPOINT 255 255 255 128 128 128 {int(_x)} {int(_y)} 16 0'
            tk.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)
            tk.sendMessage(draw_msg)

        # update the target position
        time_elapsed = flip_time - move_start
        tar_x = amp_x * sin(2 * pi * freq_x * time_elapsed + phase_x)
        tar_y = -amp_y * sin(2 * pi * freq_y * time_elapsed + phase_y)
        # break if the time elapsed exceeds the trial duration
        if time_elapsed > trial_duration:
            break
        
    # clear the screen
    clear_screen(win, bgcolor_RGB)
    tk.sendMessage('blank_screen')
    # Send a message to clear the Data Viewer screen
    tk.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    tk.stopRecording()

    # send trial variables to record in the EDF data file
    tk.sendMessage(f"!V TRIAL_VAR amp_x {amp_x:.2f}")
    tk.sendMessage(f"!V TRIAL_VAR amp_y {amp_y:.2f}") 
    tk.sendMessage(f"!V TRIAL_VAR phase_x {phase_x:.2f}") 
    pylink.pumpDelay(2) # give the tracker a break 
    tk.sendMessage(f"!V TRIAL_VAR phase_y {phase_y:.2f}") 
    tk.sendMessage(f"!V TRIAL_VAR freq_x {freq_x:.2f}")
    tk.sendMessage(f"!V TRIAL_VAR freq_y {freq_y:.2f}")
    tk.sendMessage(f"!V TRIAL_VAR mov_duration {trial_duration:.2f}")

    # send a 'TRIAL_RESULT' message to mark the end of trial
    tk.sendMessage('TRIAL_RESULT 0')


# ____________________the real experiment start here________________________
# Calibrate the tracker
# open a window for graphics and calibration
#
# request Pylink to use the Psychopy window for calibration
graphics = EyeLinkCoreGraphicsPsychoPy(tk, win)
graphics.setTargetType('circle')
pylink.openGraphicsEx(graphics)

# provide instruction and calibrate the tracker
calib_msg = visual.TextStim(win, 
    text=u'按回车键校准眼动仪', 
    font="Heiti SC", height=42)
calib_msg.draw()
win.flip()
try:
    tk.doTrackerSetup()
except RuntimeError as err:
    print('ERROR:', err)
    tk.exitCalibration()

# show the instruction sentence
intro_sentence = visual.TextStim(win, 
    text=u'屏幕中心会出现一个白色圆点\n\n请转动眼睛追随圆点的运动', 
    font="Heiti SC", height=42,pos=(0,0))
intro_sentence.draw()
win.flip()
event.waitKeys(keyList=['return'])

# present the experiment stim
random.shuffle(phi_list)
trial_id_count = 1
for phi in phi_list:
    run_a_trial(phi)
    trial_id_count += 1

# Close the EDF data file and put the tracker in idle mode
# put the tracker in offline
tk.setOfflineMode()
# wait for 100ms
pylink.pumpDelay(100)
# close the EDF data file
tk.closeDataFile()

# Downloading EDF file to a local folder ('edfData')
# show some message on screen
msg = 'Downloading EDF file from the Eyelink Host PC...'
edf = visual.TextStim(win, text=msg, color='white')
edf.draw()
win.flip()
# create a data folder if it were not there already
cwd = os.getcwd()
if not os.path.exists('edfData'):
    os.mkdir('edfData')
local_edf = os.path.join(cwd,'edfData', edf_file)
tk.receiveDataFile(edf_file, local_edf)


# Close the connection to tracker, close graphics
tk.close()
win.close()
core.quit()