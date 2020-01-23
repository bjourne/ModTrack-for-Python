from modtrack import tracker
import pygame

#################################################################
#LOAD AMIGA MOD FILE
#################################################################
ret=tracker.load_amigamodule('keys_to_imagina.mod')  # M.K. - instruments not playing or pattern in wrong order

#################################################################
#INIT, MAKE AND PLAY
#################################################################

screen = tracker.init(0x40,(100,100))
# True because we load a legacy module format ([C-2 01 F01] instead of
# [C-2 --- 01 F01 ---])
tracker.make_pattern(True)
tracker.play_pattern()


#################################################################
#PLAY WHILE USER DOES NOT QUIT PROGRAM
#################################################################

while True:#player.isAlive():
    for event in pygame.event.get():

        #Key event
        if event.type == pygame.KEYDOWN:
            #Close window
            if event.key==pygame.K_ESCAPE:
                tracker.abort_play()
                quit()

        # Window Close Button
        if event.type == pygame.QUIT:
            tracker.abort_play()
            quit()
