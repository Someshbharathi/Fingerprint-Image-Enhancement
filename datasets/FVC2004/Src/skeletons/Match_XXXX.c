/* --------------------------------------------------------------- 
            FVC match_xxxx testing program
            
            BioLab University of Bologna - Italy
                 
				 v 1.0 - October 2001

   --------------------------------------------------------------- */

#include <windows.h>
#include <stdio.h>
#include "../inc/fvc.h"

BYTE IMAGE[MAXIMAGESIZE];
int IMAGE_X,IMAGE_Y;


int Load_gray256tif(FILE* fp, BOOL Upright);


int main(int argc, char * argv[])
{   
  char imagefile[MAXPATH], templatefile[MAXPATH], configfile[MAXPATH], outputfile[MAXPATH];
  FILE *co,*im,*te,*ou;
  int err;
  BOOL Upright,MatchingPerformed;
  float similarity;
    
  // Load parameters
  if (argc!=5)
    { printf("\nSyntax error.\nUse: match_xxxx <imagefile> <templatefile> <configfile> <outputfile>\n");
	  return SYNTAX_ERROR;
	}
  strcpy(imagefile,argv[1]);
  strcpy(templatefile,argv[2]);
  strcpy(configfile,argv[3]);
  strcpy(outputfile,argv[4]);


  /* XXXX Init Library
  ....
  in case of error
    - exit returning XXXX_INIT_ERROR if your library cannot be initialized
  */
    

  /* XXXX configfile Load and Setup
  ....
  In case of error
    - exit returning CONFIG_FILE_NOT_FOUND if you cannot open the configfile
    - exit returning XXXX_SETUP_ERROR if your library cannot be configured
  In any case, remember closing the configfile
  */ 


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
    

  /* XXXX Load fingerprint template file
  .....
  In case of error
    - exit returning CANNOT_OPEN_TEMPLATE_FILE if you cannot open the templatefile
  Remember closing the templatefile
  */
  

  /* XXXX Matching
  .....
  - set MatchingPerformed=TRUE if the matching has been performed
     or MatchingPerformed=FALSE if your algorithm cannot perform the matching (e.g. insufficient quality)

  - copy into "similarity" the similarity score produced by your algorithm
    [similarity is a floating point value ranging from 0 to 1 which indicates the similarity between
	 the template and the fingerprint: 0 means no similarity, 1 maximum similarity.] 
  */

  
  /* Send the results to outputfile */
  ou=fopen(outputfile,"at");
  if (ou==NULL) return CANNOT_OPEN_OUTPUT_FILE;   
  if (fprintf(ou,"%15s %15s %4s %8.6f\n",imagefile,templatefile,MatchingPerformed?"OK":"FAIL",MatchingPerformed?similarity:0.0F)<=0)
    return CANNOT_UPDATE_OUTPUT_FILE;
  fclose(ou);


  /* XXXX Close Library
  ....
  */

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
