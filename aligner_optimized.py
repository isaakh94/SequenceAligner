#!/usr/bin/env python

import numpy as np
import argparse

PAM250 = {
'A': {'A': 2,  'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K': -1, 'L': -2, 'M': -1, 'N':  0, 'P':  1, 'Q':  0, 'R': -2, 'S':  1, 'T':  1, 'V':  0, 'W': -6, 'Y': -3},
'C': {'A': -2, 'C': 12, 'D': -5, 'E':-5, 'F': -4, 'G': -3, 'H': -3, 'I': -2, 'K': -5, 'L': -6, 'M': -5, 'N': -4, 'P': -3, 'Q': -5, 'R': -4, 'S':  0, 'T': -2, 'V': -2, 'W': -8, 'Y':  0},
'D': {'A': 0,  'C': -5, 'D':  4, 'E': 3, 'F': -6, 'G':  1, 'H':  1, 'I': -2, 'K':  0, 'L': -4, 'M': -3, 'N':  2, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
'E': {'A': 0,  'C': -5, 'D':  3, 'E': 4, 'F': -5, 'G':  0, 'H':  1, 'I': -2, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P': -1, 'Q':  2, 'R': -1, 'S':  0, 'T':  0, 'V': -2, 'W': -7, 'Y': -4},
'F': {'A': -3, 'C': -4, 'D': -6, 'E':-5, 'F':  9, 'G': -5, 'H': -2, 'I':  1, 'K': -5, 'L':  2, 'M':  0, 'N': -3, 'P': -5, 'Q': -5, 'R': -4, 'S': -3, 'T': -3, 'V': -1, 'W':  0, 'Y':  7},
'G': {'A': 1,  'C': -3, 'D':  1, 'E': 0, 'F': -5, 'G':  5, 'H': -2, 'I': -3, 'K': -2, 'L': -4, 'M': -3, 'N':  0, 'P':  0, 'Q': -1, 'R': -3, 'S':  1, 'T':  0, 'V': -1, 'W': -7, 'Y': -5},
'H': {'A': -1, 'C': -3, 'D':  1, 'E': 1, 'F': -2, 'G': -2, 'H':  6, 'I': -2, 'K':  0, 'L': -2, 'M': -2, 'N':  2, 'P':  0, 'Q':  3, 'R':  2, 'S': -1, 'T': -1, 'V': -2, 'W': -3, 'Y':  0},
'I': {'A': -1, 'C': -2, 'D': -2, 'E':-2, 'F':  1, 'G': -3, 'H': -2, 'I':  5, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -2, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -5, 'Y': -1},
'K': {'A': -1, 'C': -5, 'D':  0, 'E': 0, 'F': -5, 'G': -2, 'H':  0, 'I': -2, 'K':  5, 'L': -3, 'M':  0, 'N':  1, 'P': -1, 'Q':  1, 'R':  3, 'S':  0, 'T':  0, 'V': -2, 'W': -3, 'Y': -4},
'L': {'A': -2, 'C': -6, 'D': -4, 'E':-3, 'F':  2, 'G': -4, 'H': -2, 'I':  2, 'K': -3, 'L':  6, 'M':  4, 'N': -3, 'P': -3, 'Q': -2, 'R': -3, 'S': -3, 'T': -2, 'V':  2, 'W': -2, 'Y': -1},
'M': {'A': -1, 'C': -5, 'D': -3, 'E':-2, 'F':  0, 'G': -3, 'H': -2, 'I':  2, 'K':  0, 'L':  4, 'M':  6, 'N': -2, 'P': -2, 'Q': -1, 'R':  0, 'S': -2, 'T': -1, 'V':  2, 'W': -4, 'Y': -2},
'N': {'A': 0,  'C': -4, 'D':  2, 'E': 1, 'F': -3, 'G':  0, 'H':  2, 'I': -2, 'K':  1, 'L': -3, 'M': -2, 'N':  2, 'P':  0, 'Q':  1, 'R':  0, 'S':  1, 'T':  0, 'V': -2, 'W': -4, 'Y': -2},
'P': {'A': 1,  'C': -3, 'D': -1, 'E':-1, 'F': -5, 'G':  0, 'H':  0, 'I': -2, 'K': -1, 'L': -3, 'M': -2, 'N':  0, 'P':  6, 'Q':  0, 'R':  0, 'S':  1, 'T':  0, 'V': -1, 'W': -6, 'Y': -5},
'Q': {'A': 0,  'C': -5, 'D':  2, 'E': 2, 'F': -5, 'G': -1, 'H':  3, 'I': -2, 'K':  1, 'L': -2, 'M': -1, 'N':  1, 'P':  0, 'Q':  4, 'R':  1, 'S': -1, 'T': -1, 'V': -2, 'W': -5, 'Y': -4},
'R': {'A': -2, 'C': -4, 'D': -1, 'E':-1, 'F': -4, 'G': -3, 'H':  2, 'I': -2, 'K':  3, 'L': -3, 'M':  0, 'N':  0, 'P':  0, 'Q':  1, 'R':  6, 'S':  0, 'T': -1, 'V': -2, 'W':  2, 'Y': -4},
'S': {'A': 1,  'C':  0, 'D':  0, 'E': 0, 'F': -3, 'G':  1, 'H': -1, 'I': -1, 'K':  0, 'L': -3, 'M': -2, 'N':  1, 'P':  1, 'Q': -1, 'R':  0, 'S':  2, 'T':  1, 'V': -1, 'W': -2, 'Y': -3},
'T': {'A': 1,  'C': -2, 'D':  0, 'E': 0, 'F': -3, 'G':  0, 'H': -1, 'I':  0, 'K':  0, 'L': -2, 'M': -1, 'N':  0, 'P':  0, 'Q': -1, 'R': -1, 'S':  1, 'T':  3, 'V':  0, 'W': -5, 'Y': -3},
'V': {'A': 0,  'C': -2, 'D': -2, 'E':-2, 'F': -1, 'G': -1, 'H': -2, 'I':  4, 'K': -2, 'L':  2, 'M':  2, 'N': -2, 'P': -1, 'Q': -2, 'R': -2, 'S': -1, 'T':  0, 'V':  4, 'W': -6, 'Y': -2},
'W': {'A': -6, 'C': -8, 'D': -7, 'E':-7, 'F':  0, 'G': -7, 'H': -3, 'I': -5, 'K': -3, 'L': -2, 'M': -4, 'N': -4, 'P': -6, 'Q': -5, 'R':  2, 'S': -2, 'T': -5, 'V': -6, 'W': 17, 'Y':  0},
'Y': {'A': -3, 'C':  0, 'D': -4, 'E':-4, 'F':  7, 'G': -5, 'H':  0, 'I': -1, 'K': -4, 'L': -1, 'M': -2, 'N': -2, 'P': -5, 'Q': -4, 'R': -4, 'S': -3, 'T': -3, 'V': -2, 'W':  0, 'Y': 10}}

def needle_affine(A, B, matrix = PAM250, gap = -1, opengap = -5, EMBOSS = False, global_align = True, local_align = False):
    """
    Alignment implementation supporting both global and local alignments.
    When running with the EMBOSS flag set to True, two things change:
        - Gaps in the beginning and at the end of the sequence are not 
          subject to the "opengap" penalty.
        - The penalty for extending a gap is not added when opening one.
    EMBOSS mode only works reliably with the global alignment mode.
    """
    if local_align:
        global_align = False
    maybe_gap = True
    if EMBOSS:
        maybe_gap = False
        if not global_align:
            print('WARNING: EMBOSS mode only works reliably with global alignments.')
    #Initialising
    scores_matched = [[[-np.inf, (0, 0)]]*(len(B)+1) for i in range(len(A) + 1)]
    scores_hgap = [[[-np.inf, (0, 0)]]*(len(B)+1) for i in range(len(A) + 1)]
    scores_vgap = [[[-np.inf, (0, 0)]]*(len(B)+1) for i in range(len(A) + 1)]
    #Filling first rows and columns
    scores_matched[0][0] = [0, (0, 0)]
    for i in range(1, len(A) + 1):
        scores_hgap[i][0] = [opengap*maybe_gap + (i)*gap, (-1, 0)]
        if not global_align:
            scores_matched[i][0] = [0, (0, 0)]
    for j in range(1, len(B) + 1):
        scores_vgap[0][j] = [opengap*maybe_gap + (j)*gap, (0, -1)]
        if not global_align:
            scores_matched[0][j] = [0, (0, 0)]
    #Filling the rest of the matrix
    zero_option = -np.inf
    if not global_align:
        zero_option = 0
    for i in range(1, len(A)+1):
        for j in range(1, len(B)+1):
            options = [[matrix[A[i-1]][B[j-1]] + scores_matched[i-1][j-1][0], (-1, -1)],
                       [matrix[A[i-1]][B[j-1]] + scores_hgap[i-1][j-1][0], (-1, 0)],
                       [matrix[A[i-1]][B[j-1]] + scores_vgap[i-1][j-1][0], (0, -1)],
                       [zero_option, (0, 0)]]
            scores_matched[i][j] = max(options)
            options = [[scores_matched[i-1][j][0] + opengap + gap*maybe_gap, (-1, -1)],
                       [scores_hgap[i-1][j][0] + gap, (-1, 0)],
                       [-np.inf, (0, -1)]]
            scores_hgap[i][j] = max(options)
            options = [[scores_matched[i][j-1][0] + opengap + gap*maybe_gap, (-1, -1)],
                       [-np.inf, (-1, 0)],
                       [scores_vgap[i][j-1][0] + gap, (0, -1)]]
            scores_vgap[i][j] = max(options)
    #Backtracking
    out_A = ''
    out_B = ''
    pos = [len(A), len(B)]
    if EMBOSS:
        scores_hgap[len(A)][len(B)][0] -= opengap
        scores_vgap[len(A)][len(B)][0] -= opengap
    options = [scores_matched[len(A)][len(B)],
               scores_hgap[len(A)][len(B)],
               scores_vgap[len(A)][len(B)]]
    current_matrix = [scores_matched, scores_hgap, scores_vgap][np.argmax([i[0] for i in options])]
    if not global_align:
        current_matrix = scores_matched
        temp = np.asarray([[i[0] for i in j] for j in current_matrix])
        temp = np.unravel_index(np.argmax(temp), temp.shape)
        pos[0] = temp[0]
        pos[1] = temp[1]
    step_pointers = {(-1, -1): scores_matched, (-1, 0): scores_hgap, (0, -1): scores_vgap, (0, 0): 'END'}
    step = current_matrix[pos[0]][pos[1]]
    final_score = step[0]
    step = step[1]
    while not step == (0, 0):
        # Processing data of current position
        if not current_matrix == scores_vgap:
            out_A += A[pos[0]-1]
            pos[0] -= 1
        else:
            out_A += '-'
        if not current_matrix == scores_hgap:
            out_B += B[pos[1]-1]
            pos[1] -= 1
        else:
            out_B += '-'
        # Getting new position
        current_matrix = step_pointers[step]
        step = current_matrix[pos[0]][pos[1]][1]
    return out_A[::-1], out_B[::-1], final_score

def segment_by(string, n):
    return [string[i:i+n] for i in range(0, len(string), n)]

def format_alignment(A, B, chars_per_row = 60):
    matched = ''
    for pair in zip(A, B):
        if pair[0] == pair[1]:
            matched += ':'
        elif pair[0] == '-' or pair[1] == '-':
            matched += ' '
        else:
            matched += '.'
    if not chars_per_row:
        return '\n'.join([A, matched, B])
    newA = segment_by(A, chars_per_row)
    matched = segment_by(matched, chars_per_row)
    newB = segment_by(B, chars_per_row)
    return '\n\n'.join(['\n'.join([newA[i], matched[i], newB[i]]) for i in range(len(matched))])
        
def main():
    parser = argparse.ArgumentParser(description = 'Aligns sequence A to sequence B using affine gap penalties. If you want to align with regular gap penalties, set the --opengap flag to 0.')
    parser.add_argument('A', type = str, help = 'Sequence A')
    parser.add_argument('B', type = str, help = 'Sequence B')
    parser.add_argument('-g', '--gap', type = float, default = -1.0, help = 'Gap extension penalty (should most often be a negative number).')
    parser.add_argument('-o', '--opengap', type = float, default = -5.0, help = 'Gap opening penalty (should most often be a negative number).')
    parser.add_argument('-e', '--emboss', action = 'store_true', help = 'Use EMBOSS rules for alignment, meaning you don\'t receive gap extension when opening a gap, and having gaps at the end or beginning of a sequence does not count as opening a gap.')
    parser.add_argument('-l', '--local', action = 'store_true', help = 'Make a local alignment rather than a global.')
    parser.add_argument('-m', '--matrix', type = str, default = '', help = 'Which scoring matrix to use. The matrix should be saved as a python dictionary in plain-text. Default behaviour is using the inbuilt PAM250.')
    parser.add_argument('-c', '--charbreak', type = int, default = 0, help = 'How many characters per row you want the output to have. Default is no linebreaks.')
    args = parser.parse_args()
    
    scoring_matrix = PAM250
    if args.matrix:
        with open(args.matrix, 'r') as f:
            scoring_matrix = eval(f.read())
        assert type(scoring_matrix) is dict, 'Invalid scoring matrix format.'
    
    alignA, alignB, score = needle_affine(args.A, args.B, gap = args.gap, opengap = args.opengap, local_align = args.local, matrix = scoring_matrix, EMBOSS = args.emboss)
    
    print('SCORE: {}\n'.format(score))
    print(format_alignment(alignA, alignB, chars_per_row = args.charbreak))

if __name__ == '__main__':
    main()