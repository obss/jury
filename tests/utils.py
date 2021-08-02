import os


def assert_shell(command, exit_status=0):
    """
    Run command.

    :param command: Command plus any arguments
    :type command: str
    :param exit_status: Expected exit status
    :type exit_status: int
    :return: (exit code, standard output and error)
    :rtype: (int, str or unicode)
    :raises AssertionError: if actual exit status does not match
     exit_status
    """
    actual_exit_status = os.system(command)
    assert exit_status == actual_exit_status, "Unexpected exit code " + str(actual_exit_status)
    return actual_exit_status
