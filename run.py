#!/usr/bin/env python
import sys, os, math
import matplotlib.pyplot as plt

#-Configuration ----------------------------------------------------------------
sizes = [ 2**n for n in range(7, 14) ]
modes = ['Naive', 'SMB', 'CRB', 'CRB-T', 'CRB-TR']

#-Script Body ------------------------------------------------------------------
if __name__ == "__main__":

	#-Build the OpenCL project--------------------------------------------------
    if not 'build' in os.listdir('.'):
        os.system('mkdir build')
    os.chdir('build')

    print '# Building the OpenCL project ...'

    if (os.system('cmake ..') != 0):
        sys.exit('# Please check your OpenCL installation.')
    os.system('make')

    #-Run the Benchmark---------------------------------------------------------
    print '# Benchmark starts here ...'

    results = [[] for i in range(len(modes))]
    for i, size in enumerate(sizes):
        path = 'bench.log'
        with open(path, 'w+') as fo:
            # run the 'cpp' executable ...
            n_iteration = max(1, int((2 ** 14 / size)))
            print ('[step %s/%s] size : %4s | iter : %3s'
                  % (i+1, len(sizes), size, n_iteration))
            cmd_line = ('./GEMM -s %s -i %s -r > %s'
                       % (size, n_iteration, path))
            status = os.system(cmd_line)
            scores = fo.read()[:-1]
            # process the output ...
            if (status != 0):
                print '# Iteration failed :\n', scores
                sys.exit()
            else:
               for i, time in enumerate(scores.split('\n')):
                   tflops = (size ** 3 / float(time)) * 2 * 1e-9
                   results[i].append(tflops)

    print '# Benchmark completed !'

    #-Display the Results-------------------------------------------------------
    fig,axes = plt.subplots()

    # size and name
    fig.set_size_inches(9, 5, forward=True)
    fig.canvas.set_window_title('GEMM - benchmark')

    # axes
    axes.set_xlim([0, len(sizes)-1])
    axes.set_ylim([0, 1.2 * max([max(l) for l in results])])

    axes.xaxis.set_ticks(range(0, len(sizes)))
    axes.xaxis.set_ticklabels(sizes)

    # plotting
    def plot_entry(n):
        colors = ['r', 'g', 'b', 'm', 'k']
        markers = ['o', '^', 's', 'D', 'v']
        return plt.plot(results[n], color=colors[n],
                        linestyle='-', marker=markers[n],
                        markeredgewidth=1, markeredgecolor=colors[n],
                        markerfacecolor='none', markersize=8)[0]

    print '# Plotting the results ...'
    plots = [ plot_entry(n) for n in range(0, len(modes)) ]

    # legend
    plt.legend(plots, modes, loc='upper left')

    # background grid
    plt.grid(True, which="major", linestyle=':')
    plt.grid(True, which="minor", linestyle=':', alpha=0.25)
    plt.minorticks_on()

    # labels
    plt.xlabel('Matrix Size (M=N=K)', fontsize=13)
    plt.ylabel('Effective Performance (TFLOPS)', fontsize=13)
    plt.title('Performance Comparison of the Proposed Kernels', size=14)

    # and that's it!
    plt.show()
    print '# Exiting ...'
