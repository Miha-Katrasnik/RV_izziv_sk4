# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import my_lib as my
import time


video_file = ['videos/MVI_6342.MOV', 'videos/MVI_6339.MOV'] #vrstni red: leva roka, desna roka
threshold = 2 #prag za upragovljanje "razlike" slik
ksize = 5 #Gaussian blur kernel size
circle_r_small = 25 #radius of circle around pin
circle_r_large = 38 #outside radius of keep-out ring 
circle_th = 15 #threshold for pin inside hole
ring_th = 20 #threshold for hand inside ring (near hole)

#array za casa leve in desne roke
cas = np.zeros(2)

#vse naredimo za levo in desno roko
for i in range(2):
    
    #iz videa zajamemo prvo sliko
    cap = cv.VideoCapture(video_file[i])
    ret, frame0 = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        raise Exception("Can't receive frame (wrong filename?).")
    
    if i==0:
        print('\nLEVA ROKA')
    if i==1:
        print('\nDESNA ROKA')
    
    #določimo št. slik na sekundo
    fps = cap.get(cv.CAP_PROP_FPS)
    print('fps:', fps)
    
    #ustvarimo objekt za shranjevanje videa
    out = cv.VideoWriter('out_video_' + str(i) + '.avi',cv.VideoWriter_fourcc('M','J','P','G'), fps, (1280,720))
    
    #ustvarimo objekt za sivinsko razliko
    diffObject = my.HSDiff(frame0, threshold=2, ksize=5)
    
    
    #DOLOČITEV TOČK
    points_option = 'a' #trenutno privzeto avtomatska določitev točk
    #points_option = input('Določi nove točke? (y/n): ')
    
    if points_option == 'y':
        points = my.set_points(cv.cvtColor(frame0, cv.COLOR_BGR2RGB), n=9)
    elif points_option == 'a':
        #avtomatsko poiščemo luknje s Houghovo transformacijo
        circles = my.hough_find_holes(frame0)
        
        if len(circles) >= 9:
            points = circles[0:9,0:2]
        else:
            print('Avtomatsko določanje točk neuspešno. Prosimo določite jih ročno!')
            points = my.set_points(cv.cvtColor(frame0, cv.COLOR_BGR2RGB), n=9)
    else:
        #uporabimo vnaprej določene točke
        points = my.peg_points[i]
    
    
    #določimo maske za kroge okoli zatičev
    circle_masks_small = []
    circle_masks_large = []
    
    for point in points:
        mask = my.circle_mask((1280,720), center=point, r=circle_r_small)
        circle_masks_small.append(mask)
        
        mask = my.ring_mask((1280,720), center=point, r_min=circle_r_small, r_max=circle_r_large)
        circle_masks_large.append(mask)
    
    
    iteration = 1 #za štetje iteracije while zanke
    iter_start = 0 #spremenljivka za zap. št. slike začetka opravljanja naloge (roka pride na sceno)
    iter_stop = 0 #spremenljivka za zap. št. slike zaključka opravljanja naloge (roka zapusti sceno)
    stanje = 0 #stanje opravljanja naloge (0 - start, 1 - roka na sliki, 2 - zatiči vstavljeni, 3 - zatiči pospravljeni, 4 - roka umaknjena)
    
    time_start = time.perf_counter() #čas začetka za izračun hitrosti delovanja programa
    
    #TEST
    peg_states_prev = np.zeros(9, dtype='bool')

    max_inserted_pegs = 0 #uporabljeno pri vstavljanju zatičev
    min_inserted_pegs = 9 #uporabljeno pri razstavljanju zatičev
    pegs_inserted_time = np.zeros(9)
    pegs_extract_time = np.zeros(9)
    
    print('\nČas vstavljanja zatičev:')
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            #print("Can't receive frame (stream end?). Exiting ...")
            break
        
        #določimo razliko med trenutno in prvo sliko
        diff = diffObject.subtract(frame)
        
        #sliko razlike upragovimo
        diff_th = diff > diffObject.threshold
        diff_th = diff_th.astype('uint8') * 255
        
        #določimo, ali je roka na sliki
        diff_th_mean = np.mean(diff_th)
        hand_in_frame = diff_th_mean > 20
        
        
        #določimo sliko, ki jo želimo prikazati
        #frame_to_show = cv.cvtColor(diff_th, cv.COLOR_GRAY2BGR)
        frame_to_show = frame
        
        #narišemo okvir, ki predstavlja verjetnost, da je roka na sliki
        box_color = my.determine_color(diff_th_mean, (30,20,10,3))
        #box_color = my.determine_color(diff_th_mean, (10,8,6,4))
        cv.rectangle(frame_to_show, (5,5), (1275,715), color=box_color, thickness=3)
        
        #določanje ali je zatič v luknji
        inside_circles_arr = []
        inside_rings_arr = []
        points = np.array(points).astype('int32')
        for j in range(len(points)):
            point = points[j]
            #povprečna vrednost upragovljene slike znotraj KROGA j-tega zatiča
            inside_circle = np.mean(diff_th[ circle_masks_small[j] ])
            inside_circles_arr.append(inside_circle)
            
            #določimo barvo in narišemo krožnico
            #circle_color = my.determine_color(inside_circle, (25,15,10,5))
            circle_color = my.determine_color(inside_circle, (circle_th+3,circle_th,circle_th-1,circle_th-2))
            cv.circle(frame_to_show, center=(point[0], point[1]), radius=circle_r_small, color=circle_color, thickness=2)
            
            #povprečna vrednost upragovljene slike znotraj KOLOBARJA j-tega zatiča
            inside_ring = np.mean(diff_th[ circle_masks_large[j] ])
            inside_rings_arr.append(inside_ring)
            
            #določimo barvo in narišemo krožnico
            #ring_color = my.determine_color(inside_ring, (100,60,20,10))
            ring_color = my.determine_color(inside_ring, (ring_th+2,ring_th+1,ring_th,ring_th-1))
            cv.circle(frame_to_show, center=(point[0], point[1]), radius=circle_r_large, color=ring_color, thickness=2)
        
        #določitev stanj zatičev (vstavljen / ni vstavljen) in št. iteracije spremembe
        peg_states, peg_change_iter = my.peg_states_changes(inside_circles_arr, circle_th, inside_rings_arr, ring_th, iteration)
        
        
        #določitev časa med vstavljanjem/razstavljanjem posameznih zatičev - (dobro bi bilo napisati bolj razumljivo)
        if np.any(peg_states != peg_states_prev):
            num_of_inserted_pegs = np.sum(peg_states)
            if stanje == 1 and (num_of_inserted_pegs > max_inserted_pegs):
                #vstavljanje zatičev
                pegs_inserted_time[num_of_inserted_pegs-1] = (peg_change_iter[peg_states != peg_states_prev][0] - iter_start)/fps
                if num_of_inserted_pegs <= 1:
                    print("{0}. | {1: .2f} s".format(num_of_inserted_pegs, pegs_inserted_time[num_of_inserted_pegs-1]))
                else:    
                    print("{0}. | {1: .2f} s".format(num_of_inserted_pegs, pegs_inserted_time[num_of_inserted_pegs-1] - pegs_inserted_time[num_of_inserted_pegs-2]))
                #print('cas vstavljanja zatica:', (peg_change_iter[peg_states != peg_states_prev][0] - iter_start)/fps)
                max_inserted_pegs = num_of_inserted_pegs
            if stanje == 2 and (num_of_inserted_pegs < min_inserted_pegs):
                peg_index = 8-num_of_inserted_pegs
                pegs_extract_time[peg_index] = (peg_change_iter[peg_states != peg_states_prev][0] - iter_start)/fps
                if peg_index == 0:
                    print("{0}. | {1: .2f} s".format(9-num_of_inserted_pegs, pegs_extract_time[peg_index] - pegs_inserted_time[8]))
                else:
                    print("{0}. | {1: .2f} s".format(9-num_of_inserted_pegs, pegs_extract_time[peg_index] - pegs_extract_time[peg_index-1]))
                min_inserted_pegs = num_of_inserted_pegs
                
            peg_states_prev = peg_states.copy()
        
        
        
        #določimo trenutno stanje opravljanja naloge
        if stanje == 0:
            if diff_th_mean > 20:
                #print('Roka je na sceni')
                stanje = stanje+1
                iter_start = iteration
                
        elif stanje == 1:
            if np.all(peg_states == np.ones(9)):
                #print('Zatici so vstavljeni')
                print('\nČas razstavljanja zatičev:')
                stanje = stanje+1
        elif stanje == 2:
            if np.all(peg_states == np.zeros(9)):
                print('Zatici so pospravljeni')
                stanje = stanje+1
        elif stanje == 3:
            if diff_th_mean < 20:
                print('Roka umaknjena')
                stanje =stanje+1
                iter_stop = iteration
    
        cv.putText(frame_to_show, "Leva roka" if i==0 else "Desna roka", (50, 700), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255))
        
        #prikaz trenutnega časa opravljanja naloge na videoposnetku
        if iter_start == 0:
            time_to_show = 0
        elif iter_stop == 0:
            time_to_show = (iteration - iter_start)/fps
        else:
            time_to_show = (iter_stop - iter_start)/fps
        cv.putText(frame_to_show, "time: {0: .1f}".format(time_to_show), (50, 600), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))
        
        #besedila za različne podnaloge (bela - ni opravljeno, zelena - opravljeno)
        cv.putText(frame_to_show, "Roka prisla v kader", (10, 20), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255) if stanje<1 else (0,255,0))
        cv.putText(frame_to_show, "Vsi zatici vstavljeni (" + str(np.sum(peg_states)) + "/9)" , (10, 40), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255) if stanje<2 else (0,255,0))
        cv.putText(frame_to_show, "Vsi zatici odstranjeni", (10, 60), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255) if stanje<3 else (0,255,0))
        cv.putText(frame_to_show, "Roka zapustila kader", (10, 80), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255) if stanje<4 else (0,255,0))
        
        #prikažemo sliko
        cv.imshow('frame', frame_to_show)
        
        #shranimo sliko videa
        out.write(frame_to_show)

        
        # ZAČETEK KODE ZA FUNKCIJE TIPK
        key = cv.waitKey(1)
        
        if key == ord('w'):
            key = cv.waitKey(0)
            while key != ord('w'):
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    cv.imshow('frame', frame)
                elif key == ord('0'):
                    cv.imshow('frame', frame0.astype('uint8'))
                elif key == ord('d'):
                    diff_show = 10*diff.astype('float64')
                    diff_show[diff_show<0] = 0
                    diff_show[diff_show>255] = 255
                    diff_show = diff_show.astype('uint8')
                    cv.imshow('frame', diff_show)
                elif key == ord('t'):
                    cv.imshow('frame', diff_th)
                
                key = cv.waitKey(0)
                # uporabnik zapre okno - želimo enak rezultat kot pri pritisku na 'q'
                if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) == 0:
                    key = ord('q')
                
        if key == ord('q'):
            #print('q')
            break
        # KONEC KODE ZA FUNKCIJE TIPK
        
        
        #ko uporabnik zapre okno, prenehaj z izvajanjem while zanke
        if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) == 0:
            break
        
        
        iteration += 1
        if(iteration % 30 == 0):
            #iteration = 0
            #print('elapsed time:', time.perf_counter() - time_start)
            time_start = time.perf_counter()
    
            
    cap.release()
    out.release()
    cv.destroyAllWindows()
    
    #izračunaj čas opravljanja naloge
    if stanje == 4:
        cas[i] = (iter_stop-iter_start)/fps
    elif stanje == 3:
        print('ROKA NI ZAPUSTILA SCENE!')
    elif stanje == 2:
        print('VSI ZATIČI NISO BILI RAZSTAVLJENI!')
    elif stanje == 1:
        print('VSI ZATIČI NISO BILI VSTAVLJENI!')
    elif stanje == 0:
        print('ROKA NI PRIŠLA NA SCENO!')


#izračunaj časa opravljanja naloge (če je bila naloga opravljena pravilno)
if(cas[0] > 0):
    print('\nCas opravljanja naloge z levo roko je:', '{0: .2f}'.format(cas[0]), 's')
if(cas[1] > 0):
    print('\nCas opravljanja naloge z desno roko je:', '{0: .2f}'.format(cas[1]), 's')

if np.all(cas > 0):
    if cas[0] > cas[1]:
        print('Dominantna roka je desna.')
    else:
        print('Dominantna roka je leva.')
        
    print('Razlika časov je:', np.abs(cas[0] - cas[1]))

print('\nKonec programa')