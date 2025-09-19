#!/usr/bin/env python3
# encoding: utf-8
# The task to test the fixation of the romantic relationship
# _________________________import relevant modules__________________________
import os, random, pylink
import numpy as np
from psychopy import visual, event, core, gui, monitors
from psychopy.tools.monitorunittools import deg2pix
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


# __________________________def relevant functions__________________________
# use a dialog to record basic info of participants
def get_sub_info():
    """this is a function aiming at collecting subjects' basic information"""

    task_dlg = gui.Dlg(title=u'注视任务')
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
customMon.setSizePix((1920,1080))

# define the experiment mouse and window
win = visual.Window(fullscr=True, monitor=customMon, units='pix', screen=1)

# hide the mouse curse
win.mouseVisible = False

# set the screen resolution
SCN_W, SCN_H = win.size

# ____________________set the settings to the Eyelink_______________________
# Step 1: Connect to the tracker
tk = pylink.EyeLink('100.1.1.1')

# Step2: open an EDF data file on th EyeLink Host PC
# the file name here should match the name of the file received later
edf_file = 'fx_' + sub_info_org[0] + '_' + sub_info_org[1] + '.edf'
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

def run_a_trial(trial_index, trial_duration):
    """ Helper function specifying the events that will occur in a single trial

    arguments:
        trial_index - record the order of trial presentation in the task
        trial_duration - show the target for how long
    """
    
    bgcolor_RGB = (128, 128, 128)

    # put the tracker in the offline mode first
    tk.setOfflineMode()

    # send a "TRIALID" message to mark the start of a trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    tk.sendMessage('TRIALID %d' % trial_index)

    # record_status_message : show some info on the Host PC
    # here we show how many trial has been tested
    status_msg = 'TRIAL #: %d ' % trial_index
    tk.sendCommand("record_status_message '%s'" % status_msg)

    # drift check
    # the doDriftCorrect() function requires target position in integers
    # the last two arguments:
    # draw_target (1-default, 0-draw the target then call doDriftCorrect)
    # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
    #
    # drift-check and re-do camera setup if ESCAPE is pressed
    
    # fixation disk 
    stim_fix = visual.Circle(win, radius=23,
        fillColor='white',lineColor=False)

    # distractor disk
    stim_dis = visual.Circle(win, radius=23,
        fillColor='red',lineColor=False)
        
    # drift-check and re-do camera setup if ESCAPE is pressed
#    stim_fix.autoDraw=True
    while True:
        stim_fix.draw()
        win.flip()
        try:
            error = tk.doDriftCorrect(int(SCN_W/2.0),int(SCN_H/2.0), 0, 1)
            # break following a success drift-check
            if error is not pylink.ESC_KEY:
                break
        except:
            pass    
    
    # put tracker in idle/offline mode   before recording
    tk.setOfflineMode()
    # start recording
    pylink.pumpDelay(50)
    tk.startRecording(1, 1, 1, 1)

    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)
        
    # trial_start with a 'fixcross onset' message
    stim_fix.draw()
    win.flip()
    fix_onset_time = core.getTime()  # record the image onset time
    tk.sendMessage('fixcross_onset')
    tk.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)
    fixcross_msg = '!V FIXPOINT 255 255 255 128 128 128 %d %d 16 0' % (
        int(SCN_W/2.0), int(SCN_H/2.0))
    tk.sendMessage(fixcross_msg)

    # present the distractors
    dis_par = {"dis_rad": [3.8886726, 2.4551237, 2.1399028, 4.8736553, 0.4994609, 0.10841042, 1.1483475, 6.186778],
        "dis_dur": [1.0710722, 0.81297886, 1.2585872, 0.75649923, 1.0492529, 0.99645615, 0.7004881, 0.996883]}
    
    # set the distractor to authoDraw
    while (core.getTime() - fix_onset_time) < trial_duration:
        _time_elapsed = core.getTime() - fix_onset_time
        for i in range(8):
            dis_rad = dis_par['dis_rad'][i]
            dis_dur = dis_par['dis_dur'][i]
            # distrctor location
            r = deg2pix(2,customMon)
            dis_x = np.sin(dis_rad) * r
            dis_y = np.cos(dis_rad) * r
            stim_dis.pos = (dis_x, dis_y)

            if (_time_elapsed > (2 * (i+1) - dis_dur)) and (_time_elapsed < 2 * (i+1)):
                stim_dis.radius = 18
            else:
                stim_dis.radius = 0
            stim_fix.draw()
            stim_dis.draw()
        
        win.flip()
    
    # get rid of the fixation disk
    stim_fix.autoDraw=False

    # clear the screen
    clear_screen(win, bgcolor_RGB)
    tk.sendMessage('blank_screen')
    # Send a message to clear the Data Viewer screen
    tk.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)
    
    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    tk.stopRecording()

    # record trial variables to the EDF data file
    tk.sendMessage('!V TRIAL_VAR distractor %s' % '16sec')

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
    text=u'屏幕中心会出现一个白色圆点\n\n请一直注视这个圆点\n\n忽略红色圆点的干扰', 
    font="Heiti SC", height=42,pos=(0,0)) 
intro_sentence.draw()
win.flip()
event.waitKeys(keyList=['return'])

# we run just one trial that is 16-sec long, 
# 单独的一个呈现实验刺激的阶段
run_a_trial(1, 16.0)


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