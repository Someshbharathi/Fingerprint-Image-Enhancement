/* --------------------------------------------------------------- 
            FVC2000 match_samp: sample match_xxxx program
            
            BioLab University of Bologna - Italy
                 
   --------------------------------------------------------------- */

#include <windows.h>
#include <math.h>
#include <stdio.h>
#include "../inc/fvc2000.h"

BYTE IMAGE[MAXIMAGESIZE];
int IMAGE_X,IMAGE_Y;


int Load_gray256tif(FILE* fp, BOOL Upright);


int main(int argc, char * argv[])
{   
  char imagefile[MAXPATH], templatefile[MAXPATH], configfile[MAXPATH], outputfile[MAXPATH];
  FILE *co,*im,*te,*ou;
  int err,i;
  BOOL Upright,MatchingPerformed;
  float similarity;
  float mean_model,mean;
    
  // Load parameters
  if (argc!=5)
    { printf("\nSyntax error.\nUse: match_samp <imagefile> <templatefile> <configfile> <outputfile>\n");
	  return SYNTAX_ERROR;
	}
  strcpy(imagefile,argv[1]);
  strcpy(templatefile,argv[2]);
  strcpy(configfile,argv[3]);
  strcpy(outputfile,argv[4]);


  /* configfile Load and Setup */
  co = fopen(configfile,"rt");
  if (!co) return CANNOT_OPEN_CONFIG_FILE;
  fclose(co);


  /* Load a tif imagefile.
     The image is loaded by rows into the global array IMAGE:
	 - IMAGE_X and IMAGE_Y are the width and the height of the image respectively.
     - The generic pixel [x,y], where  x=0..IMAGE_X-1 and y=0..IMAGE_Y-1,
	     can be accessed as IMAGE[y*IMAGE_X+x]
  */
  im=fopen(imagefile,"rb");
  if (im==NULL) return CANNOT_OPEN_IMAGE_FILE;   
  Upright=TRUE; 
  // Upright=TRUE requires the image to be loaded Upright: IMAGE[0] denotes the "top-left" pixel 
  // Upright=FALSE requires the image to be loaded Upsidedown: IMAGE[0] denotes the "bottom-left" pixel 
  err=Load_gray256tif(im,Upright);
  fclose(im);
  if (err) return TIF_LOAD_ERROR;
    

  /* Load fingerprint template file */
  te = fopen(templatefile,"rb");
  if (!te) return CANNOT_OPEN_TEMPLATE_FILE;
  fread(&mean_model,4,1,te);
  fclose(te);

  /* Matching */

  mean = 0;
  for (i=0;i<IMAGE_X*IMAGE_Y;i++)
	mean += IMAGE[i];
  mean/= (IMAGE_X*IMAGE_Y);

  similarity = 1.0F - (float)(fabs(mean-mean_model) / 255);
	
  MatchingPerformed=TRUE;
  //MatchingPerformed=FALSE;


  /* Send the results to outputfile */
  ou=fopen(outputfile,"at");
  if (ou==NULL) return CANNOT_OPEN_OUTPUT_FILE;   
  if (fprintf(ou,"%s %s %s %8.6f\n",imagefile,templatefile,MatchingPerformed?"OK":"FAIL",MatchingPerformed?similarity:0.0F)<=0)
    return CANNOT_UPDATE_OUTPUT_FILE;
  fclose(ou);


  return SUCCESS;
}

/* ----------------------- */
/*    Auxiliary routines
/* ----------------------- */

BYTE buffer[512];

DWORD in_dword(DWORD i)
{ DWORD v=0;

  v=v|(buffer[i]);
  v=v|(buffer[i+1]<<8);
  v=v|(buffer[i+2]<<16);
  v=v|(buffer[i+3]<<24);
  return v;
}

WORD in_word(DWORD i)
{ WORD v=0;

  v=v|(buffer[i]);
  v=v|(buffer[i+1]<<8);
  return v;
}


// Load a 256 gray-scale uncompressed TIF image into the global array IMAGE
int Load_gray256tif(FILE* fp, BOOL Upright)
{ DWORD ifd_offset;
  WORD directory_entry_count;
  WORD offset;
  DWORD strip_offset,data_offset;
  BOOL strip_based=FALSE;
  BYTE* pimage;
  int i;

  if (fread(buffer,8,1,fp)!=1) return 1;
  if (in_word(0)!=0x4949) return 2;
  if (in_word(2)!=0x002a) return 3;
  ifd_offset=in_dword(4);
  if (fseek(fp,ifd_offset,SEEK_SET)) return 1;
  if (fread(buffer,2,1,fp)!=1) return 1;
  directory_entry_count=in_word(0);
  if (fread(buffer,directory_entry_count*12,1,fp)!=1) return 1;
  offset=0;
  while (directory_entry_count >0)
	 { switch (in_word(offset))
		 {  case 0x00fe: if (in_word(offset+8)!=0) return 4; break;
			case 0x0100: IMAGE_X = in_word(offset+8); break;
			case 0x0101: IMAGE_Y = in_word(offset+8); break;
			case 0x0102: if (in_word(offset+8)!=8) return 5; break;
			case 0x0103: if (in_word(offset+8)!=1) return 6; break;
			case 0x0106: if (in_word(offset+8)!=1) return 7; break;
			case 0x0111: strip_offset = in_word(offset+8); break;
			case 0x0115: if (in_word(offset+8)!=1) return 8; break;
			case 0x0116: if (in_word(offset+8)!= IMAGE_Y) strip_based=TRUE; break;
        	case 0x011c: if (in_word(offset+8)!=1) return 11; break;
		 }
		offset+=12;
		directory_entry_count-=1;
	 }

  if (strip_based)
    { if (fseek(fp,strip_offset,SEEK_SET)) return 1;
	  if (fread(buffer,4,1,fp)!=1) return 1;
      data_offset = in_dword(0);
    }
  else data_offset=strip_offset;
  if (fseek(fp,data_offset,SEEK_SET)) return 1;
    
  if (Upright)
    { pimage=IMAGE;
	  for (i=0;i<IMAGE_Y;i++)
	    { if (fread(pimage,IMAGE_X,1,fp)!=1) return 1;
          pimage+=IMAGE_X;
		}
    }
  else 
    { pimage=IMAGE+IMAGE_X*(IMAGE_Y-1);
	  for (i=0;i<IMAGE_Y;i++)
	    { if (fread(pimage,IMAGE_X,1,fp)!=1) return 1;
          pimage-=IMAGE_X;
		}
    }
  return 0;
}
