#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <direct.h>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main()
{
	char *buffer;
	//也可以将buffer作为输出参数
	if((buffer = getcwd(NULL, 0)) == NULL)
	{
		perror("getcwd error");
	}
	else
	{
        printf("%s\n", buffer);
		free(buffer);
	}
}