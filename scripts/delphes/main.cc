// main02.cc is a part of the PYTHIA event generator.
// Copyright (C) 2009 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This is a simple test program. It fits on one slide in a talk.
// It studies the pT_Z spectrum at the Tevatron.

#include <TROOT.h>
#include <TFile.h>
#include <TH1D.h>
#include <map>
#include <limits>       // std::numeric_limits

// ProMC file. Google does not like these warnings
#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-Wshadow"
#include "promc/ProMCBook.h"

#include "Pythia8/Pythia.h"
using namespace Pythia8;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}


string getEnvVar( std::string const & key ) {
  char * val = getenv( key.c_str() );
  return val == NULL ? std::string("") : std::string(val);
}



void readPDG( ProMCHeader * header  ) {

  string temp_string;
  istringstream curstring;


  string  PdgTableFilename  = getEnvVar("PROMC")+"/data/particle.tbl";
  if (PdgTableFilename.size()<2) {
  cout <<"**        ERROR: PROMC variable not set. Did you run source.sh" <<
         "      **" << endl;
    exit(1);
  }

  ifstream fichier_a_lire(PdgTableFilename.c_str());
  if(!fichier_a_lire.good()) {
    cout <<"**        ERROR: PDG Table ("<< PdgTableFilename
         <<  ") not found! exit.                        **" << endl;
    exit(1);
    return;
  }

  // first three lines of the file are useless
  getline(fichier_a_lire,temp_string);
  getline(fichier_a_lire,temp_string);
  getline(fichier_a_lire,temp_string);
  while (getline(fichier_a_lire,temp_string)) {
    curstring.clear(); // needed when using several times istringstream::str(string)
    curstring.str(temp_string);
    long int ID; std::string name; int charge; float mass; float width; float lifetime;
 // ID name   chg       mass    total width   lifetime
    //  1 d      -1      0.33000     0.00000   0.00000E+00
    //  in the table, the charge is in units of e+/3
    //  the total width is in GeV
    //  the lifetime is ctau in mm
    curstring >> ID >> name >> charge >> mass >> width >> lifetime;
    ProMCHeader_ParticleData* pp= header->add_particledata();
    pp->set_id(ID);
    pp->set_mass(mass);
    pp->set_name(name);
    pp->set_width(width);
    pp->set_lifetime(lifetime);
    pp->set_charge(charge);
    //cout << ID << " " << name << " " << mass << endl;
  }

}


int main(int argc, char* argv[]) {


     // Check that correct number of command-line arguments
  if (argc != 3) {
    cerr << " Unexpected number of command-line arguments. \n You are"
         << " expected to provide one input config and one output ProMC file name. \n"
         << " Program stopped! " << endl;
    return 1;
  }

   cout << "HepSim:  Pythia8 Input Configuration =" << argv[1] << endl;
   cout << "HepSim:  ProMC Output =" << argv[2] << endl;



  // Generator. Process selection. Tevatron initialization. Histogram.
  Pythia pythia;




/////////// read config files ////////////////////
    string sets="";
    string sets1="";
    bool   apply_slim=true;

    int Ntot=0;
    vector<string> configs;
    string events;
    ifstream myfile;
    myfile.open(argv[1], ios::in);
    if (!myfile) {
      cerr << "Can't open input file:  " << argv[1] << endl;
      exit(1);
    } else {
            string line;
            while(getline(myfile,line))
	  {
            //the following line trims white space from the beginning of the string
            line.erase(line.begin(), find_if(line.begin(), line.end(), not1(ptr_fun<int, int>(isspace))));
            if(line[0] == '#') continue;
            if (line.length()<3) continue;
            string tmp=string(line);
            // no empty spaces inside string
            std::string::iterator end_pos = std::remove(tmp.begin(), tmp.end(), ' ');
            tmp.erase(end_pos, tmp.end());
            bool special=false;
            int found1=tmp.find("EventsNumber");
            if (found1!=(int)std::string::npos) {events=tmp; special=true;}
            int found2=tmp.find("ApplyParticleSlim=on");
            if (found2!=(int)std::string::npos) {apply_slim=true; special=true;}
            int found3=tmp.find("ApplyParticleSlim=off");
            if (found3!=(int)std::string::npos) {apply_slim=false; special=true;}
            if (!special)  {sets1=sets1+tmp+"; "; pythia.readString(line); }
            configs.push_back(line);
            }
    myfile.close();
    vector<string> readnum=split(events,'=');
    Ntot= atoi(readnum[1].c_str());
    cout << "Reading events. " << events << " Total number is=" << Ntot<< endl;
    for (unsigned int i=0; i<configs.size(); i++) {
           cout << ".. input ="+configs[i] << endl;
           sets=sets+configs[i]+";";
    }
   } // end else
  pythia.init();

  pythia.settings.listChanged(); // Show changed settings
  double versionNumber = pythia.settings.parm("Pythia:versionNumber");
  pythia.particleData.listChanged(); // Show changed particle data
  std::stringstream s;
  s << versionNumber;
  string version=s.str();


    // book a histogram, make sure that the output name is Analysis.root
    TString  ffile("AnalysisHisto.root");
    cout << "\n -> Output file is =" << ffile << endl;
    TFile * RootFile = TFile::Open(ffile, "RECREATE", "Histogram file");
  //  TH1D * h_pt = new TH1D("ptjet", "ptjet", 200, 250, 1000);


 // ****************  book ProMC file **********************
  // ProMCBook*  epbook = new ProMCBook("Pythia8.promc","w",true);
 // no caching
 ProMCBook*  epbook = new ProMCBook(argv[2],"w");

  epbook->setDescription(Ntot,"PYTHIA-"+version+"; "+sets);
  // **************** Set a header ***************************
  ProMCHeader header;
  // cross section in pb
  header.set_cross_section( pythia.info.sigmaGen() * 1e9 );
  header.set_cross_section_error( pythia.info.sigmaErr() * 1e9 );
  // the rest
  header.set_id1( pythia.info.idA() );
  header.set_id2( pythia.info.idB() );
  header.set_pdf1( pythia.info.pdf1() );
  header.set_pdf2( pythia.info.pdf2() );
  header.set_x1(  pythia.info.x1() );
  header.set_x2(  pythia.info.x2()  );
  header.set_scalepdf(  pythia.info.QFac()  );
  header.set_weight( pythia.info.weight());
  header.set_name(sets1);  // pythia.info.name());
  header.set_code(pythia.info.code());
  header.set_ecm(pythia.info.eCM());
  header.set_s(pythia.info.s());

 // Use the range 0.01 MeV to 20 TeV using varints (integers)
 // if particle in GeV, we mutiple it by kEV, to get 0.01 MeV =1 unit
 // const double kEV=1000*100;
 // for 13 TeV, increase the precision
  double kEV=1000*100;
  double slimPT=0.3;
  // special run
  double kL=1000;

 // for 100 TeV, reduce the precision
 // const double kEV=1000*10;
 // set units dynamically
    // e+e- 250, 500 GeV
  if (pythia.info.eCM() <1000) {
        kEV=1000*1000;
        slimPT=0.1;
        kL=10000;
  }

  if (pythia.info.eCM() <20000 &&  pythia.info.eCM()>=1000) {
        kEV=1000*100;
        slimPT=0.3;
        kL=1000;

  }

  if (pythia.info.eCM() >=20000) { // larger energy, i.e. 100 TeV
        kEV=1000*10;
        slimPT=0.4;
        kL=1000;
  }

 // if lenght is in mm, use 0.1 mm = 1 unit
 // const double kL=1000*10;


  header.set_momentumunit((int)kEV);
  header.set_lengthunit((int)kL);

  cout << "HepSim: CM energy = " << pythia.info.eCM() << " GeV" << endl;
  cout << "HepSim: kEV (energy) varint unit =" << (int)kEV << endl;
  cout << "HepSim: kL (length) varint unit  =" << (int)kL << endl;
  cout << "HepSim: slimming pT = " << slimPT << " GeV" << endl;

   // let's store a map with most common masses:
  readPDG( &header );

  epbook->setHeader(header); // write header


        std::map <int,float> charges;
        for (int i=0; i<header.particledata_size(); i++){
          ProMCHeader_ParticleData p= header.particledata(i);
          string name=p.name();
          int    id=p.id();
          double charge=p.charge();
          //double mass=p.mass(); // not used
          //double width=p.width(); // not used
          //double lifetime = p.lifetime(); // not used
          //cout << "Reading PDG=" << i << " is= " << id << " name=" <<  name << " cha=" << charge/3.0 << endl;
          charges[id]=charge/3.0;
        }



  // Begin event loop. Generate event. Skip if error. List first one.
  for (int n = 0; n < Ntot; n++) {
    if (!pythia.next()) continue;
    // if (n < 1) {pythia.info.list(); pythia.event.list();}
    // Loop over particles in event. Find last Z0 copy. Fill its pT.

    if ((n<=10) ||
        ((n<=100 && (n%10) == 0)) ||
        ((n<=1000 && (n%100) == 0))  ||
        ((n>=1000 && (n%1000) == 0)) ) {
        cout << "Number of events= " << n << " passed"  << endl; };

   // If failure because reached end of file then exit event loop.
      if (pythia.info.atEndOfFile()) {
        cout << " Aborted since reached end of Les Houches Event File\n";
        break;
      }

//************  ProMC file ***************//
  ProMCEvent promc;

  // fill event information
  ProMCEvent_Event  *eve= promc.mutable_event();
  eve->set_number(n);
  eve->set_process_id(pythia.info.code());     // process ID
  eve->set_scale(pythia.info.pTHat());
  eve->set_alpha_qed(pythia.info.alphaEM());
  eve->set_alpha_qcd(pythia.info.alphaS());
  eve->set_scale_pdf(pythia.info.QFac());
  eve->set_weight(pythia.info.weight());
  eve->set_pdf1(pythia.info.weightSum() );     // special for Pythia
  eve->set_pdf2(pythia.info.mergingWeight() ); // special for Pythia
  eve->set_x1(pythia.info.x1pdf());
  eve->set_x2(pythia.info.x2pdf());
  eve->set_id1(pythia.info.id1pdf());
  eve->set_id2(pythia.info.id2pdf());


  // fill truth particle information
  ProMCEvent_Particles  *pa= promc.mutable_particles();

for (int i =0; i<pythia.event.size(); i++) {

  int pdgid=pythia.event[i].id();
  int status=pythia.event[i].statusHepMC();


  if (apply_slim) {
    int take=false;
    if (i<9) take=true;                               // first original
    if (abs(pdgid)==5 ||  abs(pdgid)==6 )             take=true; // top and b
    if (abs(pdgid)>10 && abs(pdgid)<17)               take=true; // leptons etc.
    if (abs(pdgid)>22 && abs(pdgid)<37)               take=true; // exotic
    if (status ==1 && pythia.event[i].pT()>slimPT)    take=true; // final state
    if (take==false) continue;
  }


  double ee=pythia.event[i].e()*kEV;
  double px=pythia.event[i].px()*kEV;
  double py=pythia.event[i].py()*kEV;
  double pz=pythia.event[i].pz()*kEV;
  double mm=pythia.event[i].m()*kEV;
  double xx=pythia.event[i].xProd()*kL;
  double yy=pythia.event[i].yProd()*kL;
  double zz=pythia.event[i].zProd()*kL;
  double tt=pythia.event[i].tProd()*kL;

  //if (pythia.event[i].tProd()>100) cout << "Time is " << pythia.event[i].tProd() << endl;

/* just a check. do we truncate energy?
  double maxval=2147483647; // std::numeric_limits<int>::min()
  double minval=0.5;
  bool  err=false;
  if (abs(px)>=maxval || abs(py)>=maxval || abs(pz)>= maxval ||
      abs(ee)>=maxval || abs(mm)>=maxval || abs(xx)>= maxval ||
      abs(yy)>=maxval || abs(zz)>=maxval || abs(tt)>= maxval) err=true;
  if (err){
          cout << "Event =" << i << " Value is too large for varint. Change units: " << kEV << " or " << kL << endl;
          cout << ee << " " << px << " " << pz << " " << ee << " " << mm << " " << xx << " " << yy << " " << zz << " " << tt << endl;
          exit(1);
          }

   err=false;
    if ((abs(px)<minval && abs(px)>0) ||
        (abs(py)<minval && abs(py)>0) ||
        (abs(pz)<minval && abs(pz)>0) ||
        (abs(ee)<minval && abs(ee)>0) ||
        (abs(mm)<minval && abs(mm)>0) ||
        (abs(xx)<minval && abs(xx)>0) ||
        (abs(yy)<minval && abs(yy)>0) ||
        (abs(zz)<minval && abs(zz)>0) ||
        (abs(tt)<minval && abs(tt)>0) ) err=true;
    if (err){
          //cout << "Event =" << i << " Value is too small for varint. Change units: kEV=" << kEV << " kL=" << kL << endl;
          //cout << ee << " " << px << " " << pz << " " << ee << " " << mm << " " << xx << " " << yy << " " << zz << " " << tt << endl;
          //exit(1);
          }
*/

  pa->add_pdg_id( pdgid );
  pa->add_status(  status );
  pa->add_px( (int)px );
  pa->add_py( (int)py );
  pa->add_pz( (int)pz  );
  pa->add_mass( (int)mm );
  pa->add_energy( (int)ee );
  pa->add_mother1( pythia.event[i].mother1()  );
  pa->add_mother2( pythia.event[i].mother2()  );
  pa->add_daughter1( pythia.event[i].daughter1()  );
  pa->add_daughter2( pythia.event[i].daughter2()   );
  pa->add_barcode( 0 ); // dummy
  pa->add_weight( 1 ); // dummy
  pa->add_charge( charges[pdgid]  ); // dummy
  pa->add_id( i  );
  pa->add_x( (int)xx  );
  pa->add_y( (int)yy  );
  pa->add_z( (int)zz  );
  pa->add_t( (int)tt  );

 }

  epbook->write(promc); // write event




  } // endl loop over events


   // To check which changes have actually taken effect
   pythia.settings.listChanged();
   // pythia.particleData.listChanged();
   pythia.particleData.list(25);
   // ParticleDataTable::listAll()
   // ParticleDataTable::list(25);


   pythia.stat();


  // Output histograms
  double sigmapb = pythia.info.sigmaGen() * 1.0E9;
  double sigmapb_err = pythia.info.sigmaErr() * 1.0E9;

  cout << "== Run statistics: " << endl;
  cout << "== Cross section    =" <<  sigmapb << " +- " << sigmapb_err << " pb" << endl;
  cout << "== Generated Events =" <<  Ntot << endl;
  double lumi=(Ntot/sigmapb)/1000;
  cout << "== Luminosity       =" <<  lumi  << " fb-1" << endl;
  cout << "\n\n-- Output file=" << ffile << endl;
  cout << "\n\n";

    RootFile->Write();
    RootFile->Print();
    RootFile->Close();


// save post-generation statistics for ProMC
  ProMCStat stat;
  stat.set_cross_section_accumulated( sigmapb ); // in pb
  stat.set_cross_section_error_accumulated( pythia.info.sigmaErr() * 1e9 );
  stat.set_luminosity_accumulated(  Ntot/sigmapb );
  stat.set_ntried(pythia.info.nTried());
  stat.set_nselected(pythia.info.nSelected());
  stat.set_naccepted(pythia.info.nAccepted());
  epbook->setStatistics(stat);

  // close the ProMC file
  epbook->close(); // close


  return 0;
}
