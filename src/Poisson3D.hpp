#ifndef POISSON_1D_HPP
#define POISSON_1D_HPP

#include <deal.II/base/timer.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson3D
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;
class Sfera : public Function<dim>
  {
   public:
    //Costruttore
    Sfera(Point<dim> &p, double r)
    : centro(p)
    , raggio(r)
    {}

    bool appartenenza(const Point<dim> &d)
    {
      Point<dim> diff;
      
      for (unsigned int i = 0; i < dim; i++)
        diff[i] = d[i] - centro[i];
      double distance = diff.norm();

      if( distance <= raggio)
        return true;
      else
        return false;

      //return 0; 
    }

    bool non_overlapping(const Point<dim> &a)
    {
       Point<dim> diff;
      
      for (unsigned int i = 0; i < dim; i++)
        diff[i] = a[i] - centro[i];
      double distance = diff.norm();

      if( distance > 2*raggio)
        return true;
      else
        return false;
    } 

    protected:
    // centro della sfera
    const Point<dim> centro;

    // raggio della sfera
    const double raggio;

  };

  // Diffusion coefficient.
  // In deal.ii, functions are implemented by deriving the dealii::Function
  // class, which provides an interface for the computation of function values
  // and their derivatives.
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    DiffusionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // Value of epsilon
      const double eps = 3.0;

      // Creation of spheres
      Point<dim> center;
      center[0] = 0.85;
      center[1] = 0.85;
      center[2] = 0.85;
      const double rax = 0.1;
      Sfera sfera(center, rax);

      
      Point<dim> center2;
      center2[0] = 0.15;
      center2[1] = 0.15;
      center2[2] = 0.15;
      const double rax2 = 0.1;
      Sfera sfera2(center2, rax2);

      Point<dim> center3;
      center3[0] = 0.85;
      center3[1] = 0.15;
      center3[2] = 0.15;
      const double rax3 = 0.1;
      Sfera sfera3(center3, rax3);


      Point<dim> center4;
      center4[0] = 0.15;
      center4[1] = 0.85;
      center4[2] = 0.85;
      const double rax4 = 0.1;
      Sfera sfera4(center4, rax4);

      Point<dim> center5;
      center5[0] = 0.85;
      center5[1] = 0.85;
      center5[2] = 0.15;
      const double rax5 = 0.1;
      Sfera sfera5(center5, rax5);
      
      Point<dim> center6;
      center6[0] = 0.15;
      center6[1] = 0.15;
      center6[2] = 0.85;
      const double rax6 = 0.1;
      Sfera sfera6(center6, rax6);
      
      Point<dim> center7;
      center7[0] = 0.85;
      center7[1] = 0.15;
      center7[2] = 0.85;
      const double rax7 = 0.1;
      Sfera sfera7(center7, rax7);

      Point<dim> center8;
      center8[0] = 0.15;
      center8[1] = 0.85;
      center8[2] = 0.15;
      const double rax8 = 0.1;
      Sfera sfera8(center8, rax8);

       Point<dim> center9;
      center9[0] = 0.50;
      center9[1] = 0.15;
      center9[2] = 0.50;
      const double rax9 = 0.1;
      Sfera sfera9(center9, rax9);

       Point<dim> center10;
      center10[0] = 0.50;
      center10[1] = 0.85;
      center10[2] = 0.50;
      const double rax10 = 0.1;
      Sfera sfera10(center10, rax10);

      Point<dim> center11;
      center11[0] = 0.15;
      center11[1] = 0.50;
      center11[2] = 0.50;
      const double rax11 = 0.1;
      Sfera sfera11(center11, rax11);

      Point<dim> center12;
      center12[0] = 0.85;
      center12[1] = 0.50;
      center12[2] = 0.50;
      const double rax12 = 0.1;
      Sfera sfera12(center12, rax12);

      Point<dim> center13;
      center13[0] = 0.50;
      center13[1] = 0.15;
      center13[2] = 0.15;
      const double rax13 = 0.1;
      Sfera sfera13(center13, rax13);

      Point<dim> center14;
      center14[0] = 0.50;
      center14[1] = 0.85;
      center14[2] = 0.15;
      const double rax14 = 0.1;
      Sfera sfera14(center14, rax14);

      Point<dim> center15;
      center15[0] = 0.50;
      center15[1] = 0.85;
      center15[2] = 0.85;
      const double rax15 = 0.1;
      Sfera sfera15(center15, rax15);


      Point<dim> center16;
      center16[0] = 0.50;
      center16[1] = 0.15;
      center16[2] = 0.85;
      const double rax16 = 0.1;
      Sfera sfera16(center16, rax16);

      Point<dim> center17;
      center17[0] = 0.50;
      center17[1] = 0.50;
      center17[2] = 0.50;
      const double rax17 = 0.2;
      Sfera sfera17(center17, rax17);

    // Flag di appartenenza
     bool flag = sfera.appartenenza(p);
     bool flag2 = sfera2.appartenenza(p);
     bool flag3 = sfera3.appartenenza(p);
     bool flag4 = sfera4.appartenenza(p);
     bool flag5 = sfera5.appartenenza(p);
     bool flag6 = sfera6.appartenenza(p);
     bool flag7 = sfera7.appartenenza(p);
     bool flag8 = sfera8.appartenenza(p);
     bool flag9 = sfera9.appartenenza(p);
     bool flag10 = sfera10.appartenenza(p);
     bool flag11 = sfera11.appartenenza(p);
     bool flag12 = sfera12.appartenenza(p);
     bool flag13 = sfera13.appartenenza(p);
     bool flag14 = sfera14.appartenenza(p);
     bool flag15 = sfera15.appartenenza(p);
     bool flag16 = sfera16.appartenenza(p);
     bool flag17 = sfera17.appartenenza(p);
     


    // Valutation of mu
      if (flag == true ||
          flag2 == true ||
          flag3 == true ||
          flag4 == true ||
          flag5 == true ||
          flag6 == true ||
          flag7 == true ||
          flag8 == true ||
          flag9 == true ||
          flag10 == true ||
          flag11 == true ||
          flag12 == true ||
          flag13 == true ||
          flag14 == true ||
          flag15 == true ||
          flag16 == true ||
          flag17 == true
          )
        return std::pow(10.0, eps);
      else if (!flag)
        return 0.05;  

      return 0.0;  



    }
    
  };

  // Reaction coefficient.
  class ReactionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    ReactionCoefficient()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };

  // Forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    // Constructor.
    ForcingTerm()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };

  // Dirichlet boundary conditions.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Constructor.
  Poisson3D(const unsigned int &N_, const unsigned int &r_)
    : N(N_)
    , r(r_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
    , computing_timer(MPI_COMM_WORLD,
                    pcout,
                    TimerOutput::never,
                    TimerOutput::wall_times)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

protected:
  // N+1 is the number of elements.
  const unsigned int N;

  // Polynomial degree.
  const unsigned int r;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Diffusion coefficient.
  DiffusionCoefficient diffusion_coefficient;

  // Reaction coefficient.
  ReactionCoefficient reaction_coefficient;

  // Forcing term.
  ForcingTerm forcing_term;

  // g(x).
  FunctionG function_g;

  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a triangulation that is completely distributed (i.e. each process only
  // knows about the elements it owns and its ghost elements).
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  TrilinosWrappers::MPI::Vector solution;

  // Parallel output stream.
  ConditionalOStream pcout;

  TimerOutput computing_timer;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;
};

#endif