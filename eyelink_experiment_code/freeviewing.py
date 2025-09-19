#!/usr/bin/env python3
# encoding: utf-8
# The task to test the free viewing of the romantic relationship
# import relevant modules

import os

# _________________________import relevant modules__________________________
import pylink
from psychopy import visual, event, core, gui, monitors

from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy


# __________________________def relevant functions__________________________
# use a dialog to record basic info of participants
def get_sub_info():
    """this is a function aiming at collecting subjects' basic information"""

    task_dlg = gui.Dlg(title=u'图片浏览任务')
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
edf_file = 'fv_' + sub_info_org[0] + '_' + sub_info_org[1] + '.edf'
print(edf_file)
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
tk.sendCommand(f"screen_pixel_coords = 0 0 {SCN_W - 1} {SCN_H - 1}")

# Record a DISPLAY_SCREEN message to let Data Viewer know the
# correct screen resolution to use when visualizing the data
tk.sendMessage(f"DISPLAY_COORDS = 0 0 {SCN_W - 1} {SCN_H - 1}")

# Set the calibration type to 9-point (HV9)
tk.sendCommand("calibration_type = HV9")

# Set calibration and validation area 
tk.sendCommand("validation_area_proportion = 0.65 0.90")
tk.sendCommand("calibration_area_proportion = 0.65 0.90")


# ___________________define the function of a single trial__________________

def run_a_trial(trial_index, trial_image):
    """ Helper function specifying the events that will occur in a single trial

    arguments:
        trial_index - record the order of trial presentation in the task
        trial_image - image to show to the participants 
        {'top':'face_1.jpg', 'bottom':'face_2.jpg']
    """

    bgcolor_RGB = (128, 128, 128)

    # put the tracker in the offline mode first
    tk.setOfflineMode()

    # send a "TRIALID" message to mark the start of a trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    tk.sendMessage('TRIALID %d' % trial_index)

    # record_status_message : show some info on the Host PC
    # here we show how many trial has been tested
    status_msg = 'TRIAL #: %d - images: %s, %s' % (trial_index, trial_image['top'], trial_image['bottom'])
    tk.sendCommand("record_status_message '%s'" % status_msg)

    # drift check
    # the doDriftCorrect() function requires target position in integers
    # the last two arguments:
    # draw_target (1-default, 0-draw the target then call doDriftCorrect)
    # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
    #
    # drift-check and re-do camera setup if ESCAPE is pressed

    stim_fix = visual.TextStim(win, text='+', height=36, anchorHoriz='center')
    stim_fix.draw()
    win.flip()
    _fix_on = core.getTime()

    #    # drift-check and re-do camera setup if ESCAPE is pressed
    #    while True:
    #        stim_fix.draw()
    #        win.flip()
    #        try:
    #            error = tk.doDriftCorrect(int(SCN_W/2.0),int(SCN_H/2.0), 0, 1)
    #            # break following a success drift-check
    #            if error is not pylink.ESC_KEY:
    #                break
    #        except:
    #            pass

    # put tracker in idle/offline mode before recording
    tk.setOfflineMode()
    # start recording
    pylink.pumpDelay(50)
    tk.startRecording(1, 1, 1, 1)

    # Allocate some time for the tracker to cache some samples
    while True:
        if core.getTime() - _fix_on >= 1.0:
            break

    # the above pumpDelay help to present the fixation for 1-sec

    # show the image, and log a message to mark the onset of the image
    # load the image
    im_w, im_h = 409, 306
    pos_top = 0, 237
    pos_bot = 0, -237
    cwd = os.getcwd()
    _top = os.path.join(cwd, 'images', trial_image['top'])
    _bot = os.path.join(cwd, 'images', trial_image['bottom'])
    _img_top = visual.ImageStim(win, image=_top, size=[im_w, im_h], pos=pos_top)
    _img_bot = visual.ImageStim(win, image=_bot, size=[im_w, im_h], pos=pos_bot)
    _img_top.draw()
    _img_bot.draw()
    win.flip()
    img_onset_time = core.getTime()  # record the image onset time
    tk.sendMessage('image_onset')

    # send over a message to specify IAs for the images stored relative
    # to the EDF data file, see Data Viewer User Manual, "Protocol for
    # EyeLink Data to Viewer Integration" 

    # aoi for the top image
    _top = int(SCN_H / 2 - pos_top[1] - im_h / 2)
    _bottom = int(SCN_H / 2 - pos_top[1] + im_h / 2)
    _left = int(SCN_W / 2 - im_w / 2)
    _right = int(SCN_W / 2 + im_w / 2)
    ia_top = (1, _left, _top, _right, _bottom, 'im_top')
    tk.sendMessage('!V IAREA RECTANGLE %d %d %d %d %d %s' % ia_top)

    # aoi for the bottom image
    _top = int(SCN_H / 2 - pos_bot[1] - im_h / 2)
    _bottom = int(SCN_H / 2 - pos_bot[1] + im_h / 2)
    ia_bot = (2, _left, _top, _right, _bottom, 'im_bot')
    tk.sendMessage('!V IAREA RECTANGLE %d %d %d %d %d %s' % ia_bot)

    # background image for Data Viewing visualization
    top_image = '../images/' + trial_image['top']
    bot_image = '../images/' + trial_image['bottom']
    top_msg = '!V IMGLOAD CENTER %s %d %d %d %d' % (top_image,
                                                    int(SCN_W / 2.0), int(SCN_H / 2.0 - pos_top[1]), im_w, im_h)
    bot_msg = '!V IMGLOAD CENTER %s %d %d %d %d' % (bot_image,
                                                    int(SCN_W / 2.0), int(SCN_H / 2.0 - pos_bot[1]), im_w, im_h)
    tk.sendMessage(top_msg)
    tk.sendMessage(bot_msg)

    # each picture present 5000ms
    while (core.getTime() - img_onset_time) < 5.0:
        pass

    # clear the screen
    clear_screen(win, bgcolor_RGB)
    tk.sendMessage('blank_screen')

    # Send a message to clear the Data Viewer screen
    tk.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    tk.stopRecording()

    # record trial variables to the EDF data file
    tk.sendMessage('!V TRIAL_VAR img_top %s' % trial_image['top'])
    tk.sendMessage('!V TRIAL_VAR img_bot %s' % trial_image['bottom'])

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
                                 text=u'屏幕上会依次呈现一组图片\n\n请自由观察这些图片',
                                 font="Heiti SC", height=42, pos=(0, 0))
intro_sentence.draw()
win.flip()
event.waitKeys(keyList=['return'])

# ———————————————————load images ___________________________________________

images = [{"top": "im_15.jpg", "bottom": "im_16.jpg"},
          {"top": "im_23.jpg", "bottom": "im_24.jpg"},
          {"top": "im_29.jpg", "bottom": "im_30.jpg"},
          {"top": "im_13.jpg", "bottom": "im_14.jpg"},
          {"top": "im_1.jpg", "bottom": "im_2.jpg"},
          {"top": "im_25.jpg", "bottom": "im_26.jpg"},
          {"top": "im_39.jpg", "bottom": "im_40.jpg"},
          {"top": "im_17.jpg", "bottom": "im_18.jpg"},
          {"top": "im_11.jpg", "bottom": "im_12.jpg"},
          {"top": "im_19.jpg", "bottom": "im_20.jpg"},
          {"top": "im_33.jpg", "bottom": "im_34.jpg"},
          {"top": "im_3.jpg", "bottom": "im_4.jpg"},
          {"top": "im_21.jpg", "bottom": "im_22.jpg"},
          {"top": "im_37.jpg", "bottom": "im_38.jpg"},
          {"top": "im_7.jpg", "bottom": "im_8.jpg"},
          {"top": "im_27.jpg", "bottom": "im_28.jpg"},
          {"top": "im_9.jpg", "bottom": "im_10.jpg"},
          {"top": "im_5.jpg", "bottom": "im_6.jpg"},
          {"top": "im_35.jpg", "bottom": "im_36.jpg"},
          {"top": "im_31.jpg", "bottom": "im_32.jpg"}]

# present the experimental stim
for trial_index, trial_image in enumerate(images):
    run_a_trial(trial_index, trial_image)

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
local_edf = os.path.join(cwd, 'edfData', edf_file)
tk.receiveDataFile(edf_file, local_edf)

# Close the connection to tracker, close graphics
tk.close()
win.close()
core.quit()
