"""
Class to create multiple DFT simulations from a
:class:`morty.atomistic.CellModeller` instance.

"""
import os

__all__ = ['DFTCaller']


class DFTCaller:
    """
    Class to create DFT calculation files for CASTEP and GAUSSIAN.

    Used with a rasterized parameter setup from
    :class:`morty.atomistic.CellModeller` to create a series of folders
    with all input files required to run a DFT calculation.

    """
    def __init__(self, cell_modeller):
        """
        Instantiates a DFTCaller class.

        Parameters
        ----------
        cell_modeller : :class:`morty.atomistic.CellModeller`
            This instance should be set up with all parameters
            by calling
            :class:`morty.atomistic.CellModeller.rasterize_setup()`.

        """
        self.cell_modeller = cell_modeller

    def rasterize(self, foldername='.', jobname='raster', program='castep',
                  update_submitall=None, queue_template=None):
        """
        Create calculation folders for the parameter range set in
        :class:`morty.atomistic.CellModeller`.

        Parameters
        ----------
        update_submitall : list of [function, params, bool]
            A function to define, which calculation has finished, e.g. checking
            for the existance of a '.magres' file in a certain folder.
            Is used in the same way as *testfunction*, but accepts only one
            function. Only triggers which steps are considered in the queue
            file submitting all steps.
        foldername : str
            The name of the folder holding all the calculations. Defaults to
            ".".
        jobname: str
            The name of the job.
        program : str
            Holds the type of calculation to be performed. Can be one of
            ['gaussian', 'castep'].
        queue_template : str
            If provided, a queue submission file will be created. See
            :class:`morty.atomistic.Cell.set_up_job()` for details.

        Returns
        -------
        acc : list of int
            The list with the indices of the accepted steps, and the updated
            steps. Updated steps are created, when the raster is run on a
            folder already containing finished calculations. This criterion is
            governed by *update_submitall*.

        """
        updated_steps = []
        submit_all_string = str()
        if os.path.isdir(foldername) is not True:
            os.mkdir(foldername)
        for i in range(len(self.cell_modeller)):
            self.cell_modeller[i].properties['stepnum'] = i
            self.cell_modeller[i].cellname = jobname + '_' + str(i)

            self.cell_modeller[i].set_up_job(
                calcfolder=self.cell_modeller[i].foldername,
                jobname=self.cell_modeller[i].cellname,
                program=program,
                queue_template=queue_template)
            if update_submitall is None:
                submit_all_string += str('cd ' + str(i) + '; qsub ' +
                                         jobname + "_" + str(i) +
                                         '_qsub.sh; cd ..;\n')
                updated_steps.append(i)
            elif (update_submitall[0](self.cell_modeller[i],
                                      *(update_submitall[1])) is
                  update_submitall[2]):
                submit_all_string += str('cd ' + str(i) + '; qsub ' +
                                         jobname + "_" + str(i) +
                                         '_qsub.sh; cd ..;\n')
                updated_steps.append(i)

        if queue_template is not None:
            with open(os.path.join(foldername, 'submitusall.sh'),
                      'w') as submitall_file:
                submitall_file.write(submit_all_string)

        return True
