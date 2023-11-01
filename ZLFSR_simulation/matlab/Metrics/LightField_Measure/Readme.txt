============================================================================================
The light field quality measure
Copyright(c) 2020 Xiongkuo Min, Jiantao Zhou, Guangtao Zhai, Patrick Le Callet, 
Xiaokang Yang, and Xinping Guan
All Rights Reserved.

--------------------------------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation for educational
and research purposes only and without fee is hereby granted, provided that this copyright
notice and the original authors' names appear on all copies and supporting documentation.
This software shall not be used, redistributed, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the authors. The authors
make no representations about the suitability of this software for any purpose. It is
provided "as is" without express or implied warranty.
--------------------------------------------------------------------------------------------

This is the light field quality measure described in the following paper:

Xiongkuo Min, Jiantao Zhou, Guangtao Zhai, Patrick Le Callet, Xiaokang Yang, and 
Xinping Guan, "A Metric for Light Field Reconstruction, Compression, and Display Quality 
Evaluation," IEEE Transactions on Image Processing, vol. 29, pp. 3790-3804, 2020.

Please contact Xiongkuo Min (minxiongkuo@gmail.com) if you have any questions.

--------------------------------------------------------------------------------------------

Demo code:
demo.m

The light field quality measure:
function score =  LightField_Measure(View_RefName,View_DisName)
Input:  (1) View_RefName: file names of the reference views
        (2) View_DisName: file names of the distorted views
Output: (1) score: quality score
Usage:  Given the file names of the reference and distorted views
        score =  LightField_Measure(View_RefName,View_DisName)

============================================================================================
